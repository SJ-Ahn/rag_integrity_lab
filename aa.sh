export OPENAI_API_KEY=${OPENAI_API_KEY}
python evaluation/generate_golden.py --count 30 --version v3 --log-level INFO
