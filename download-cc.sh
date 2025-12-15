CRAWL=CC-MAIN-2025-47
# ln -s '/Volumes/My Passport/datasets/commoncrawl' ./data/commoncrawl
OUTDIR=./data/commoncrawl/${CRAWL}
N=5
mkdir -p "$OUTDIR"

# about 70M per file
URLS=$(curl -L "https://data.commoncrawl.org/crawl-data/${CRAWL}/wet.paths.gz" \
  | gunzip -c \
  | head -n "$N" \
  | sed 's#^#https://data.commoncrawl.org/#')

echo "Will download ($N):"
echo "$URLS"

echo "$URLS" | while read -r url; do
  wget -c -P "$OUTDIR" "$url"
done
