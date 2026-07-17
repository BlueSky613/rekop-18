# 풀이 A — 공격형 (max benchmark) · 2026-07-16

## 전략
**벤치 최대치에 베팅.** 전량 데이터 + 고용량 앙상블(9모델) + payload 미러링 + mild recency(반감기10).

## 검증 결과 (실제 scoring.py, 홀드아웃 마지막 3일)
```
comp=0.9405  AP=0.9735  R@5%=0.8326  safety=1.00  hfpr=0.000
배치263 몰수0  최저0.799   ← 작은창까지 몰수 0 (믿음성 확인)
```

## 구성 (모든 검증된 원칙 적용)
| 원칙 | 적용 |
|---|---|
| payload 미러링 학습 | ✅ train/serve 스큐 0 |
| top-K 안전캡(작은창 최소2) | ✅ 몰수 0 |
| recency 가중(반감기 10) | ✅ |
| 매니페스트 안 보냄 | ✅ -0.10 벌점 회피 |
| 비복제(자체코드) | ✅ originality 벌점 회피 |
| 9모델 앙상블 | 5×LGBM(용량↑) + RF + ExtraTrees + HistGBM + LogReg |

## ★정직한 성능 고지
- **벤치마크(연습) 0.94는 라이브 예측치 아님.** 라이브(숨은 eval)는 어려워 1등도 0.55. **A의 실전 기대 = ~0.55 무리권.**
- **보장되는 것**: 몰수 0 + 스큐 0 + 매니페스트/복제 벌점 0. (손해요인 제거)
- **천장 돌파는 라이브에서만 확인** — 벤치로는 89와 대등.

## 배포
```bash
python train.py --all      # 전체데이터로 최종 학습 (배포 직전)
python verify.py           # 몰수0 자가확인
# 미너 실행 (서브넷 repo 안, axon 포트/도달 필수, 매니페스트 안 보냄)
POKER44_MY_UID=<UID> POKER44_REPORT_URL=http://<대시보드>:8127 python miner.py --axon.port <PORT> ...
POKER44_MY_UID=<UID> python daily_update.py   # 일일 재학습 cron
```

## 언제 A를 쓰나
벤치 랭킹이 라이브에 잘 전이된다는 쪽에 베팅. **152(또는 새 UID)에 A 배포** 권장 → B와 실전 대조.

자세한 근거: [../2026-07-16_LEARNINGS.md](../2026-07-16_LEARNINGS.md)
