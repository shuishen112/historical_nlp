This repository is for historical newspaper analysis.

## Temporal Word Embeddings with Compass (TWEC)

TWEC is a temporal word embeddings model that uses a compass to align embeddings across different time periods.

Other parts are coming soon.

## collect the historical newspaper data

### Preview what would be downloaded (one year)
```bash
python3 download_mediastream_pdfs.py --years 1816 --dry-run
```

### Download the historical newspaper data
```bash
python3 download_mediastream_pdfs.py --years 1816
```

### Download several years
```bash
python3 download_mediastream_pdfs.py --years 1816,1820-1825
```
### Options
```bash
--paper-id — defaults to your URL’s ID
--output-dir -- where PDFs are saved (default: mediastream_pdfs)
--delay 0.5 -- pause between requests (be polite to the server)
--no-skip-existing -- re-download files that already exist
```