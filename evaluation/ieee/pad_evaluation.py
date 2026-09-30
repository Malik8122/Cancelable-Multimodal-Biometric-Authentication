"""Part 15: presentation-attack evaluation framework (ISO/IEC 30107-3 terminology).

The shipped system has NO presentation-attack-detection (PAD) subsystem, so APCER / BPCER (error rates of a PAD
classifier) are undefined for it. The applicable metric is the IMPOSTOR ATTACK PRESENTATION MATCH RATE (IAPMR):
the fraction of attack presentations of a target that the biometric comparator accepts as that target.

Expected data (consented, anonymous), mirroring evaluation/real_user_evaluation.py's enrollment layout:
    data/pad/enrolled/<P001>/enroll/...                       (same layout as the real-user harness)
    data/pad/attacks/<species>/<P001>/<any>.png|.wav          species: face_print, face_screen_replay,
                                                              face_video_replay, voice_replay, voice_cloning
Run: python -m evaluation.ieee.pad_evaluation
Without data the output states NOT AVAILABLE for every species - no number is produced.
"""

from __future__ import annotations

from pathlib import Path

from evaluation.ieee.common import DATA, FUTURE, REAL, RESULTS, write_csv

SPECIES = {"face_print": "face", "face_screen_replay": "face", "face_video_replay": "face", "voice_replay": "voice",
           "voice_cloning": "voice"}
ROOT = DATA / "pad"


def run() -> list[dict]:
    attacks = ROOT / "attacks"
    if not (ROOT / "enrolled").exists() or not attacks.exists():
        rows = [{"species": s, "modality": m, "IAPMR": "NOT AVAILABLE", "APCER": "NOT APPLICABLE (no PAD subsystem)",
                 "BPCER": "NOT APPLICABLE (no PAD subsystem)", "evidence_label": FUTURE,
                 "note": f"no attack data under {ROOT.relative_to(DATA.parent)}"} for s, m in SPECIES.items()]
        write_csv(RESULTS / "pad_results.csv", rows, FUTURE)
        return rows

    from evaluation.real_user_evaluation import APP, _rgb, _wav, enroll_participants

    db, services, _, enrolled, _ = enroll_participants(ROOT / "enrolled")
    rows = []
    for species, modality in SPECIES.items():
        folder = attacks / species
        accepted = total = fta = 0
        if folder.exists():
            for target_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
                target = target_dir.name
                if modality not in enrolled.get(target, set()):
                    continue
                for f in sorted(target_dir.iterdir()):
                    try:
                        emb = services[modality].embed(_wav(f) if modality == "voice" else _rgb(f))
                    except Exception:  # noqa: BLE001 - attack not even acquired
                        fta += 1
                        continue
                    total += 1
                    accepted += int(services[modality].authenticate_embedding(db, emb, target, APP).authenticated)
        rows.append({"species": species, "modality": modality, "attack_presentations": total, "failures_to_acquire": fta,
                     "accepted_as_target": accepted,
                     "IAPMR": accepted / total if total else "NOT AVAILABLE",
                     "APCER": "NOT APPLICABLE (no PAD subsystem)", "BPCER": "NOT APPLICABLE (no PAD subsystem)",
                     "evidence_label": REAL if total else FUTURE})
    write_csv(RESULTS / "pad_results.csv", rows, REAL)
    return rows

if __name__ == "__main__":
    for r in run():
        print(r)
