import numpy as np
import torch
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from collections import defaultdict
import re
from typing import Dict, List, Tuple, Optional
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class TWECFramework:
    """
    Temporal Word Embeddings with a Compass (TWEC)
    
    Two-stage approach:
    1. Train independent Word2Vec models for each time period (slices)
    2. Align embeddings using a compass (initialization) and learn transformations
    """
    
    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 5,
        workers: int = 4,
        epochs: int = 10,
        sg: int = 1,  # 1 for skip-gram, 0 for CBOW
        negative: int = 5,
        compass_method: str = 'procrustes'
    ):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.sg = sg
        self.negative = negative
        self.compass_method = compass_method
        
        # Store models and data
        self.slice_models = {}
        self.aligned_models = {}
        self.compass_vocab = set()
        self.documents_by_year = defaultdict(list)
        self.transformation_matrices = {}
        
    def extract_year_from_filename(self, filename: str) -> str:
        """Extract year from filename"""
        year = filename.split('/')[-1].split('-')[0]
        return year
    
    def preprocess_text(self, text: str) -> List[str]:
        """Clean and tokenize historical text"""
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\-áéíóúñüÃÃÃÃÃ]', ' ', text.lower())
        text = re.sub(r'\s+', ' ', text)
        
        # Simple tokenization
        tokens = text.split()
        
        # Filter
        tokens = [token for token in tokens if len(token) > 2 and not token.isdigit()]
        
        return tokens
    
    def add_document(self, filename: str, content: str):
        """Add document to appropriate time slice"""
        year = self.extract_year_from_filename(filename)
        if year:
            tokens = self.preprocess_text(content)
            if tokens:
                self.documents_by_year[year].append(tokens)
                
    def split_into_sentences(self, documents: List[List[str]], max_length: int = 100) -> List[List[str]]:
        """
        Split long documents into sentence-like chunks for better training
        """
        sentences = []
        for doc in documents:
            # Split into chunks if document is too long
            if len(doc) > max_length:
                for i in range(0, len(doc), max_length):
                    chunk = doc[i:i + max_length]
                    if len(chunk) >= 5:  # Minimum sentence length
                        sentences.append(chunk)
            else:
                if len(doc) >= 5:
                    sentences.append(doc)
        return sentences
    
    def train_slice_models(self):
        """
        Stage 1: Train independent Word2Vec model for each time slice
        FIXED: Properly build vocabulary before training
        """
        print("\n" + "="*60)
        print("STAGE 1: Training Independent Slice Models")
        print("="*60)
        
        for year in sorted(self.documents_by_year.keys()):
            documents = self.documents_by_year[year]
            print(f"\nTraining model for year {year}")
            print(f"  Documents: {len(documents)}")
            
            # Split into sentences for better training
            sentences = self.split_into_sentences(documents)
            print(f"  Sentences/chunks: {len(sentences)}")
            
            total_tokens = sum(len(sent) for sent in sentences)
            print(f"  Total tokens: {total_tokens}")
            
            if total_tokens < 100:
                print(f"  ⚠️  WARNING: Very few tokens in {year}. Model may be unreliable.")
            
            try:
                # Method 1: Standard initialization (RECOMMENDED)
                model = Word2Vec(
                    sentences=sentences,
                    vector_size=self.vector_size,
                    window=self.window,
                    min_count=self.min_count,
                    workers=self.workers,
                    epochs=self.epochs,
                    sg=self.sg,
                    negative=self.negative,
                    seed=42  # For reproducibility
                )
                
                # Alternative Method 2: Build vocabulary explicitly
                # Uncomment if Method 1 fails
                """
                model = Word2Vec(
                    vector_size=self.vector_size,
                    window=self.window,
                    min_count=self.min_count,
                    workers=self.workers,
                    sg=self.sg,
                    negative=self.negative,
                    seed=42
                )
                
                # Build vocabulary
                print("  Building vocabulary...")
                model.build_vocab(sentences)
                
                # Train
                print("  Training...")
                model.train(
                    sentences,
                    total_examples=model.corpus_count,
                    epochs=self.epochs
                )
                """
                
                self.slice_models[year] = model
                print(f"  ✓ Vocabulary size: {len(model.wv)}")
                
                # Show sample words
                if len(model.wv) > 0:
                    sample_words = list(model.wv.index_to_key[:5])
                    print(f"  Sample words: {sample_words}")
                
            except Exception as e:
                print(f"  ✗ Error training model for {year}: {e}")
                continue
    
    def train_slice_models_incremental(self):
        """
        Alternative training method with incremental vocabulary building
        Use this if standard method fails
        """
        print("\n" + "="*60)
        print("STAGE 1: Training Slice Models (Incremental)")
        print("="*60)
        
        for year in sorted(self.documents_by_year.keys()):
            documents = self.documents_by_year[year]
            print(f"\nTraining model for year {year}")
            
            sentences = self.split_into_sentences(documents)
            print(f"  Sentences: {len(sentences)}")
            
            # Initialize model
            model = Word2Vec(
                vector_size=self.vector_size,
                window=self.window,
                min_count=self.min_count,
                workers=self.workers,
                sg=self.sg,
                seed=42
            )
            
            # Build vocabulary from corpus
            print("  Building vocabulary...")
            model.build_vocab(corpus_iterable=sentences, progress_per=1000)
            print(f"  Vocabulary size: {len(model.wv)}")
            
            # Train model
            print("  Training...")
            model.train(
                corpus_iterable=sentences,
                total_examples=len(sentences),
                epochs=self.epochs,
                report_delay=1.0
            )
            
            self.slice_models[year] = model
            print(f"  ✓ Model trained successfully")
    
    def identify_compass_words(
        self, 
        min_freq_ratio: float = 0.8,
        max_compass_words: int = 500
    ) -> set:
        """
        Identify stable words across time periods (compass words)
        """
        print("\n" + "="*60)
        print("Identifying Compass Words")
        print("="*60)
        
        if len(self.slice_models) < 2:
            print("⚠️  Need at least 2 time slices for compass identification")
            return set()
        
        # Get vocabulary for each time slice
        vocabularies = {}
        for year, model in self.slice_models.items():
            if len(model.wv) > 0:
                vocabularies[year] = set(model.wv.index_to_key)
        
        if len(vocabularies) < 2:
            print("⚠️  Not enough valid vocabularies")
            return set()
        
        # Find words appearing in all slices
        common_words = set.intersection(*vocabularies.values())
        print(f"Words appearing in all time periods: {len(common_words)}")
        
        if len(common_words) == 0:
            print("⚠️  No common words found across all time periods")
            print("   This usually means your documents are too small or too different")
            print("   Falling back to words in most periods...")
            
            # Fallback: use words appearing in at least N-1 periods
            word_counts = defaultdict(int)
            for vocab in vocabularies.values():
                for word in vocab:
                    word_counts[word] += 1
            
            n_periods = len(vocabularies)
            common_words = {
                word for word, count in word_counts.items() 
                if count >= max(2, n_periods - 1)
            }
            print(f"Words appearing in {max(2, n_periods - 1)}+ periods: {len(common_words)}")
        
        if len(common_words) < 10:
            print("⚠️  Very few common words. Results may be unreliable.")
            print("   Consider: 1) Adding more documents, 2) Lowering min_count, 3) Better preprocessing")
        
        # Filter by frequency stability
        compass_candidates = []
        
        for word in common_words:
            # Get frequency rank in each slice
            ranks = []
            freqs = []
            
            for year, model in self.slice_models.items():
                if word in model.wv:
                    vocab = model.wv.key_to_index
                    rank = vocab[word]
                    freq = model.wv.get_vecattr(word, "count")
                    ranks.append(rank)
                    freqs.append(freq)
            
            if len(ranks) < 2:
                continue
            
            # Check stability
            rank_std = np.std(ranks)
            mean_rank = np.mean(ranks)
            mean_freq = np.mean(freqs)
            
            # Coefficient of variation
            if mean_rank > 0:
                cv = rank_std / mean_rank
                if cv < 0.5:  # Stable frequency
                    compass_candidates.append((word, mean_rank, cv, mean_freq))
        
        # Sort by frequency (prefer common words)
        compass_candidates.sort(key=lambda x: x[3], reverse=True)
        
        # Select top N compass words
        n_compass = min(max_compass_words, len(compass_candidates))
        self.compass_vocab = {word for word, _, _, _ in compass_candidates[:n_compass]}
        
        print(f"✓ Selected {len(self.compass_vocab)} compass words")
        if len(self.compass_vocab) > 0:
            sample = list(self.compass_vocab)[:10]
            print(f"  Example compass words: {sample}")
        
        return self.compass_vocab
    
    def procrustes_alignment(
        self,
        source_vectors: np.ndarray,
        target_vectors: np.ndarray
    ) -> np.ndarray:
        """
        Orthogonal Procrustes alignment
        """
        # Center the vectors
        source_mean = source_vectors.mean(axis=0)
        target_mean = target_vectors.mean(axis=0)
        
        source_centered = source_vectors - source_mean
        target_centered = target_vectors - target_mean
        
        # SVD of cross-covariance matrix
        M = target_centered.T @ source_centered
        U, _, Vt = np.linalg.svd(M)
        
        # Optimal rotation matrix
        Q = U @ Vt
        
        return Q
    
    def align_embeddings_procrustes(self, reference_year: Optional[str] = None):
        """
        Stage 2: Align embeddings using Procrustes alignment
        """
        print("\n" + "="*60)
        print("STAGE 2: Aligning Embeddings (Procrustes)")
        print("="*60)
        
        if len(self.slice_models) < 2:
            print("⚠️  Need at least 2 time slices for alignment")
            return
        
        if len(self.compass_vocab) < 10:
            print("⚠️  Too few compass words. Alignment may be unreliable.")
        
        years = sorted(self.slice_models.keys())
        
        # Use first year as reference
        if reference_year is None:
            reference_year = years[0]
        
        print(f"Reference year: {reference_year}")
        reference_model = self.slice_models[reference_year]
        
        # Get reference compass embeddings
        ref_compass_words = [w for w in self.compass_vocab 
                            if w in reference_model.wv]
        
        if len(ref_compass_words) < 5:
            print(f"⚠️  Only {len(ref_compass_words)} compass words in reference. Using all available.")
            ref_compass_words = list(reference_model.wv.index_to_key[:100])
        
        ref_compass_vectors = np.array([
            reference_model.wv[w] for w in ref_compass_words
        ])
        
        print(f"Using {len(ref_compass_words)} compass words for alignment")
        
        # Align each time slice
        for year in years:
            print(f"\nAligning {year} to {reference_year}")
            
            if year == reference_year:
                # Reference remains unchanged
                self.aligned_models[year] = {
                    word: reference_model.wv[word].copy()
                    for word in reference_model.wv.index_to_key
                }
                self.transformation_matrices[year] = np.eye(self.vector_size)
                print("  ✓ Reference (no transformation)")
            else:
                model = self.slice_models[year]
                
                # Get compass embeddings for this slice
                slice_compass_words = [w for w in ref_compass_words 
                                      if w in model.wv]
                
                if len(slice_compass_words) < 5:
                    print(f"  ⚠️  Only {len(slice_compass_words)} common compass words")
                
                slice_compass_vectors = np.array([
                    model.wv[w] for w in slice_compass_words
                ])
                
                # Get corresponding reference vectors
                ref_subset = np.array([
                    reference_model.wv[w] for w in slice_compass_words
                ])
                
                # Compute alignment matrix
                Q = self.procrustes_alignment(
                    slice_compass_vectors,
                    ref_subset
                )
                
                self.transformation_matrices[year] = Q
                
                # Apply transformation to all words
                self.aligned_models[year] = {}
                for word in model.wv.index_to_key:
                    original_vector = model.wv[word]
                    aligned_vector = original_vector @ Q
                    self.aligned_models[year][word] = aligned_vector
                
                # Compute alignment quality
                aligned_compass = slice_compass_vectors @ Q
                alignment_error = np.mean(
                    np.linalg.norm(aligned_compass - ref_subset, axis=1)
                )
                print(f"  ✓ Alignment error: {alignment_error:.4f}")
                print(f"  ✓ Vocabulary size: {len(self.aligned_models[year])}")
    
    def compute_semantic_shift(
        self,
        word: str,
        year1: str,
        year2: str,
        metric: str = 'cosine'
    ) -> Optional[float]:
        """
        Compute semantic shift between two time periods
        """
        if (year1 not in self.aligned_models or 
            year2 not in self.aligned_models):
            return None
        
        if (word not in self.aligned_models[year1] or 
            word not in self.aligned_models[year2]):
            return None
        
        vec1 = self.aligned_models[year1][word]
        vec2 = self.aligned_models[year2][word]
        
        if metric == 'cosine':
            similarity = np.dot(vec1, vec2) / (
                np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8
            )
            return float(1 - similarity)
        
        elif metric == 'euclidean':
            return float(np.linalg.norm(vec1 - vec2))
        
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def get_temporal_trajectory(
        self,
        word: str,
        normalize_vectors: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Get word's embedding trajectory across time
        """
        trajectory = {}
        
        for year in sorted(self.aligned_models.keys()):
            if word in self.aligned_models[year]:
                vec = self.aligned_models[year][word].copy()
                if normalize_vectors:
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec = vec / norm
                trajectory[year] = vec
        
        return trajectory
    
    def detect_semantic_change(
        self,
        words: List[str],
        threshold: float = 0.3
    ) -> Dict[str, List[Tuple[str, str, float]]]:
        """
        Detect significant semantic changes
        """
        results = {}
        years = sorted(self.aligned_models.keys())
        
        for word in words:
            changes = []
            
            for i in range(len(years) - 1):
                year1, year2 = years[i], years[i + 1]
                shift = self.compute_semantic_shift(word, year1, year2)
                
                if shift is not None and shift > threshold:
                    changes.append((year1, year2, shift))
            
            if changes:
                results[word] = changes
        
        return results
    
    def get_nearest_neighbors(
        self,
        word: str,
        year: str,
        k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Get k nearest neighbors in given year
        """
        if year not in self.aligned_models or word not in self.aligned_models[year]:
            return []
        
        target_vec = self.aligned_models[year][word]
        target_norm = np.linalg.norm(target_vec)
        
        if target_norm == 0:
            return []
        
        # Compute similarities
        similarities = []
        for other_word, other_vec in self.aligned_models[year].items():
            if other_word != word:
                other_norm = np.linalg.norm(other_vec)
                if other_norm > 0:
                    sim = np.dot(target_vec, other_vec) / (target_norm * other_norm)
                    similarities.append((other_word, float(sim)))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:k]
    
    def save(self, filepath: str):
        """Save model"""
        data = {
            'config': {
                'vector_size': self.vector_size,
                'window': self.window,
                'min_count': self.min_count,
                'compass_method': self.compass_method
            },
            'aligned_models': self.aligned_models,
            'compass_vocab': self.compass_vocab,
            'transformation_matrices': self.transformation_matrices
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✓ Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls(**data['config'])
        instance.aligned_models = data['aligned_models']
        instance.compass_vocab = data['compass_vocab']
        instance.transformation_matrices = data['transformation_matrices']
        
        return instance


class TWECVisualizer:
    """Visualization tools for TWEC"""
    
    @staticmethod
    def plot_semantic_trajectory_2d(
        twec: TWECFramework,
        word: str,
        figsize: Tuple[int, int] = (12, 8)
    ):
        """Plot 2D trajectory"""
        trajectory = twec.get_temporal_trajectory(word)
        
        if len(trajectory) < 2:
            print(f"⚠️  Word '{word}' appears in {len(trajectory)} time period(s) only")
            return None
        
        years = sorted(trajectory.keys())
        vectors = np.array([trajectory[year] for year in years])
        
        # PCA
        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(vectors)
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Trajectory line
        ax.plot(coords_2d[:, 0], coords_2d[:, 1], 
                'o-', linewidth=2, markersize=10, alpha=0.7, color='steelblue')
        
        # Year labels
        for i, year in enumerate(years):
            ax.annotate(
                year,
                (coords_2d[i, 0], coords_2d[i, 1]),
                xytext=(8, 8),
                textcoords='offset points',
                fontsize=11,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7)
            )
        
        # Arrows
        for i in range(len(coords_2d) - 1):
            arrow = FancyArrowPatch(
                coords_2d[i],
                coords_2d[i + 1],
                arrowstyle='->,head_width=0.4,head_length=0.8',
                color='red',
                alpha=0.6,
                linewidth=2.5
            )
            ax.add_patch(arrow)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
        ax.set_title(f'Semantic Trajectory: "{word}"', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_semantic_shift_matrix(
        twec: TWECFramework,
        words: List[str],
        figsize: Tuple[int, int] = (14, 10)
    ):
        """Plot shift heatmap"""
        years = sorted(twec.aligned_models.keys())
        n_years = len(years)
        
        # Compute shifts
        shift_matrix = np.zeros((len(words), n_years - 1))
        
        for i, word in enumerate(words):
            for j in range(n_years - 1):
                shift = twec.compute_semantic_shift(
                    word, years[j], years[j + 1]
                )
                shift_matrix[i, j] = shift if shift is not None else np.nan
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(shift_matrix, cmap='YlOrRd', aspect='auto', vmin=0)
        
        ax.set_xticks(range(n_years - 1))
        ax.set_xticklabels([f"{years[i]}→{years[i+1]}" 
                           for i in range(n_years - 1)], rotation=45, ha='right')
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Semantic Shift', rotation=270, labelpad=20)
        
        # Add values
        for i in range(len(words)):
            for j in range(n_years - 1):
                if not np.isnan(shift_matrix[i, j]):
                    ax.text(j, i, f'{shift_matrix[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=9)
        
        ax.set_title('Semantic Shift Matrix', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        return fig


# ==================== USAGE EXAMPLE ====================

def main():
    print("🚀 TWEC Framework - Temporal Word Embeddings with Compass\n")
    
    # Initialize
    twec = TWECFramework(
        vector_size=100,
        window=5,
        min_count=3,  # Lower for small datasets
        workers=4,
        epochs=15,
        sg=1  # Skip-gram
    )
    
    # Load documents (use your actual file paths)
    print("📁 Loading documents...\n")
    
    # Example: Load from files
    import glob
    
    for filepath in glob.glob("text_input/*.txt"):
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                twec.add_document(filepath, content)
                print(f"  ✓ Loaded: {filepath}")
        except Exception as e:
            print(f"  ✗ Error loading {filepath}: {e}")
    # Check data
    print(f"\n📊 Data summary:")
    for year in sorted(twec.documents_by_year.keys()):
        n_docs = len(twec.documents_by_year[year])
        n_tokens = sum(len(doc) for doc in twec.documents_by_year[year])
        print(f"  {year}: {n_docs} documents, {n_tokens} tokens")
    
    # Train
    print("\n" + "="*60)
    twec.train_slice_models()
    
    # Identify compass
    twec.identify_compass_words(max_compass_words=300)
    
    # Align
    twec.align_embeddings_procrustes()
    
    # Save
    twec.save('twec_model.pkl')
    
    # Analysis
    if len(twec.aligned_models) >= 2:
        print("\n" + "="*60)
        print("SEMANTIC ANALYSIS")
        print("="*60)
        
        # Get some common words
        all_words = set()
        for year_vocab in twec.aligned_models.values():
            all_words.update(year_vocab.keys())
        
        # Sample analysis
        target_words = list(all_words)[:10]
        
        for word in target_words:
            trajectory = twec.get_temporal_trajectory(word)
            if len(trajectory) >= 2:
                print(f"\n{word}:")
                years = sorted(trajectory.keys())
                for i in range(len(years) - 1):
                    shift = twec.compute_semantic_shift(word, years[i], years[i+1])
                    if shift:
                        print(f"  {years[i]}→{years[i+1]}: {shift:.4f}")
        
        # Visualize
        visualizer = TWECVisualizer()
        
        for word in target_words[:3]:
            fig = visualizer.plot_semantic_trajectory_2d(twec, word)
            if fig:
                fig.savefig(f'trajectory_{word}.png', dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✓ Saved trajectory_{word}.png")
    
    print("\n✅ Complete!")


if __name__ == "__main__":
    main()