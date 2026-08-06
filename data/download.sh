#!/usr/bin/env bash
# Download all data files for the letter-valence analysis.
# Idempotent: skips files that already exist.
set -e

DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DATA_DIR"

UA="Mozilla/5.0 (X11; Linux x86_64) research-script"

download() {
    local url="$1"
    local out="$2"
    if [ -s "$out" ]; then
        size=$(stat -c%s "$out" 2>/dev/null || stat -f%z "$out")
        echo "  [skip] $out already present (${size} bytes)"
        return 0
    fi
    echo "  [get ] $url"
    echo "        -> $out"
    if ! curl -fsSL --max-time 120 -A "$UA" "$url" -o "$out.tmp"; then
        echo "  [FAIL] curl failed for $url"
        rm -f "$out.tmp"
        return 1
    fi
    mv "$out.tmp" "$out"
    size=$(stat -c%s "$out" 2>/dev/null || stat -f%z "$out")
    echo "  [ok  ] $out (${size} bytes)"
}

echo "=== Downloading data files ==="

# 1. Warriner 2013 affective norms
download \
    "https://github.com/JULIELab/XANEW/raw/master/Ratings_Warriner_et_al.csv" \
    "Ratings_Warriner_et_al.csv"

# 2. FinancialPhraseBank (binary 50Agree split from a maintained mirror)
download \
    "https://raw.githubusercontent.com/seandearnaley/sentiment_data_sets/master/data/inputs/FinancialPhraseBank-v1.0/Sentences_50Agree.txt" \
    "Sentences_50Agree.txt"

# 3. English word list (for building letter frequency table)
download \
    "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt" \
    "words_alpha.txt"

# 4. CMU Pronouncing Dictionary (for phonetic features)
download \
    "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict" \
    "cmudict.dict"

echo ""
echo "=== Building derived files ==="

# 5. Letter frequencies from words_alpha.txt (derived)
if [ -s "letter_freqs.json" ]; then
    echo "  [skip] letter_freqs.json already present"
else
    echo "  [get ] letter_freqs.json (deriving from words_alpha.txt)"
    python3 -c "
import sys
sys.path.insert(0, '$(dirname "$0")/../src')
from data import build_letter_frequencies
freqs = build_letter_frequencies('words_alpha.txt')
import json
with open('letter_freqs.json', 'w') as f:
    json.dump(freqs, f)
print('  Wrote letter_freqs.json')
"
fi

# 6. articles_binary.csv (derived from Sentences_50Agree.txt)
if [ -s "articles_binary.csv" ]; then
    echo "  [skip] articles_binary.csv already present"
else
    echo "  [get ] articles_binary.csv (deriving from Sentences_50Agree.txt)"
    python3 -c "
import sys
sys.path.insert(0, '$(dirname "$0")/../src')
from data import build_articles_binary
df = build_articles_binary('Sentences_50Agree.txt')
df.to_csv('articles_binary.csv', index=False)
print('  Wrote articles_binary.csv with', len(df), 'rows')
"
fi

echo ""
echo "=== Done ==="
echo "Files in $DATA_DIR:"
ls -la "$DATA_DIR" | grep -v '^total'
