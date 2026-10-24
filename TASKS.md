# Tasks - Review II is Tuesday 2026-10-27

A = Manikanta, B = teammate. Tick boxes in your PR. Anything unfinished by Monday night moves to the Final Review list at the bottom.

## Saturday night (Oct 24)
- [ ] A: create the private GitHub repo, push this folder, add B as collaborator
- [ ] B: accept invite, clone, `pip install -r requirements.txt`, put the CSV in `data/raw/`
- [ ] A+B: run `test_modules.py`, `data_prep.py`, `baselines.py`, `train_anfis.py` on BOTH machines
- [ ] A: send Claude `reports/comparison.csv` and `reports/anfis_ra_grid.csv`
- [ ] B: confirm your numbers equal A's (same `comparison.csv`). Different = investigate first

## Sunday (Oct 25)
| Owner | Task | Rubric item |
|---|---|---|
| A | Literature table, 6-8 papers, reuse the verified citations from the Phase I proposal; refine research gap in 3 sentences | Lit survey (3) |
| A | Fill the article skeleton once Claude drafts it (title, abstract, methodology, results, target venue) | Article (4) |
| B | Run `python audit_odd_cow.py`: find the 1 cow whose label changes across days; decide keep or drop; log it in JOURNAL | Data prep (3) |
| B | Open the Mendeley dataset page; confirm what I*/E* sensor columns mean; write one line in `docs/HANDOFF.md` section 5 | Data prep (3) |
| B | Read `anfis.py` + `subtractive_clustering.py` line by line; prepare a 15-min explanation for A | SC justification (2), Viva (2) |
| Claude | Architecture + data-flow diagrams, article draft, slides v1 with speaking notes | System design (4), Results (4) |

## Monday (Oct 26)
- [ ] A explains data_prep + split + baselines to B (15 min). B explains clustering + ANFIS to A (15 min). Swap questions.
- [ ] Review slides together; every number traced to a file in `reports/`
- [ ] Viva drill: each person answers 5 questions out loud (Claude provides them), tag the repo `review2`
- [ ] Freeze: no new features after Monday 6 pm

## Tuesday (Oct 27): Review II

## After Review II (Final Review list)
- [ ] GA tuning of cluster radii (ra, rb)
- [ ] Grouped 5-fold CV, 2+ competing algorithms, significance test, MasPA comparison
- [ ] Ablation of the Asymmetry Index formula (e.g. coefficient of variation)
- [ ] Streamlit demo (`app.py`)
- [ ] Submission-ready paper, patent and startup notes, full report
