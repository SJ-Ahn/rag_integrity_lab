#!/usr/bin/env bash
set -euo pipefail

echo "🧹 RAG 스토어 및 로그 초기화 중..."

# 데이터 및 로그 디렉토리 정리
# data/source는 원본 데이터이므로 삭제하지 않음
rm -rf data/working/*
rm -rf data/index/*
rm -rf logs/*

# 디렉토리 다시 생성 (gitkeep 역할)
mkdir -p data/working
mkdir -p data/index
mkdir -p logs

echo "✅ 초기화 완료. 새로운 테스트를 시작할 준비가 되었습니다."
