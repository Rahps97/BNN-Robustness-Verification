# Minting the Zenodo DOI for this repository

Springer Nature's code policy requires a permanent identifier for the code behind a
paper: *"providing a GitHub link only is not sufficient as it does not assign a
permanent identifier to the code."* This file is the checklist for whoever holds the
Zenodo account. Everything that can be prepared in advance already is — the metadata
Zenodo will read lives in [`../.zenodo.json`](../.zenodo.json), and the citation
metadata GitHub renders lives in [`../CITATION.cff`](../CITATION.cff).

Nothing here has been done yet. No Zenodo account has been created, linked or
authenticated by the preparation work.

---

## 0. Before you do anything: three decisions

| Decision | Recommendation | Why it cannot wait |
| --- | --- | --- |
| **Which commit gets the DOI** | the merged `main` after PR #1, released as `v1.0.0` — see [§4](#4-which-version-to-archive) | the archive is immutable once published |
| **Is the licence really MIT?** | confirm with all three authors *before* publishing | `LICENSE` is inside the archive; changing it later needs a whole new version |
| **Who publishes** | the upstream owner (`Rahps97`) — see [§1](#1-the-github-release-route-preferred) | a fork's release would archive the fork, not the canonical repository |

The `LICENSE` file currently says MIT with copyright "2026 Rahul Singh, Seyran Saeedi,
Zheng Zhang". That was added during the reproducibility work as a reasonable default,
**not** on the authors' instruction. Zenodo lets you edit a published record's licence
*metadata*, but the `LICENSE` file baked into the archived ZIP is permanent, so the two
would then disagree forever. Settle it first.

---

## 1. The GitHub-release route (preferred)

This mints the DOI automatically and links it to the release, with no manual metadata
entry at all. **It must be done on the upstream repository,
`https://github.com/Rahps97/BNN-Robustness-Verification`, by an account with admin
rights on it** (i.e. `Rahps97`). Releasing from the `seyrans` fork would archive the
fork under a DOI that points at a non-canonical copy, and the paper would then cite the
wrong artefact.

1. **Merge PR #1** into upstream `main`. The release must contain the reproducibility
   fixes — that is the whole point of archiving `main` rather than the tag (§4).

2. **Link Zenodo to GitHub.** Go to <https://zenodo.org>, *Log in with GitHub* (or, if
   the Zenodo account already exists, *Settings → Linked accounts → GitHub → Connect*).
   Grant the requested repository access.

3. **Switch the repository on.** <https://zenodo.org/account/settings/github/> lists the
   repositories you administer. Flip the toggle next to
   `Rahps97/BNN-Robustness-Verification` to **On**. This installs a webhook.

   > Zenodo only archives releases created **after** the toggle is on. It will not pick
   > up the existing `paper-results-v1` tag, and a git tag on its own is not a release.

4. **Cut the release.** On GitHub: *Releases → Draft a new release*.
   - Tag: `v1.0.0` (create it on `main`).
   - Target: `main`.
   - Title: `v1.0.0 - archived release for the npj Unconventional Computing paper`.
   - Body: a few lines is enough; Zenodo takes its description from `.zenodo.json`, not
     from the release notes.
   - **Publish release.**

5. **Wait ~1 minute.** Zenodo downloads the release ZIP, reads `.zenodo.json` and
   publishes the record automatically. Two DOIs appear:
   - a **version DOI** (e.g. `10.5281/zenodo.1234568`) — this exact snapshot;
   - a **concept DOI** (e.g. `10.5281/zenodo.1234567`) — always resolves to the newest
     version.

   Both are shown on the record page; the concept DOI is the one labelled *"Cite all
   versions"*.

6. **Fix the one thing that cannot be automated.** Open the record → *Edit* → *Funding*
   and add the DOE award by hand:
   - Funder: **United States Department of Energy** (ROR `01bj3aw27`)
   - Award number: **DE-SC0021323**, as a custom award.

   NSF grant 2311295 is already there: it is in Zenodo's award vocabulary
   (`021nxhr62::2311295`) and `.zenodo.json` declares it. DE-SC0021323 is not in that
   vocabulary, so it cannot be declared in the file — putting it there would make the
   whole release fail. It is recorded in the record's description and notes instead.

   Save. Metadata edits after publication do not change the DOI.

7. **Sanity-check the record**: three creators in the right order, MIT licence,
   `isSupplementTo → 10.48550/arXiv.2602.13536`, keywords, and the four `data/`
   archives present inside the ZIP.

### If the toggle does not list the repository

Zenodo caches the repository list. Click *Sync now* on the GitHub settings page. If it
still does not appear, the account lacks admin rights on the upstream repository — that
is the case that §2 exists for.

---

## 2. The manual-upload fallback

Use this only if the upstream owner cannot run §1. The DOI is just as permanent; what
you lose is the automatic link to the GitHub release and the automatic archiving of
future releases.

A ready-to-upload snapshot (~10 MB gzipped) was produced alongside this checklist and
handed over separately. It is a complete clone — the working tree **plus the full git
history and all tags** — so `paper-results-v1`, its commit `47b877e` and every commit
message survive *inside* the archive. That is strictly more provenance than GitHub's
release ZIP, which contains no `.git` directory at all, and it is the one respect in
which the manual route beats the automatic one.

If you no longer have that file, this is exactly how it was made:

```bash
git clone --no-local /path/to/repo /tmp/zenodo-snapshot/BNN-Robustness-Verification
git -C /tmp/zenodo-snapshot/BNN-Robustness-Verification checkout main
tar czf /tmp/zenodo-snapshot/BNN-Robustness-Verification.tar.gz \
    -C /tmp/zenodo-snapshot BNN-Robustness-Verification
```

Then either:

**(a) Through the web form.** <https://zenodo.org/uploads/new> → drop the tarball in →
set *Resource type* to **Software** → fill the fields to match `.zenodo.json`
(title, description, the three creators with affiliations, MIT, keywords, the arXiv
related identifier as *is supplement to*, and both awards) → *Publish*.

**(b) Through the API, which reuses `.zenodo.json` verbatim.** Create a personal access
token at <https://zenodo.org/account/settings/applications/tokens/new/> with the
`deposit:write` and `deposit:actions` scopes, then:

```bash
TOKEN=...   # your Zenodo personal access token
JSON=.zenodo.json
TARBALL=/path/to/BNN-Robustness-Verification.tar.gz

# 1. create the deposition, wrapping .zenodo.json in the required {"metadata": ...}
python3 -c 'import json,sys; print(json.dumps({"metadata": json.load(open(sys.argv[1]))}))' "$JSON" \
  > /tmp/deposit.json
DEP=$(curl -s -H "Content-Type: application/json" \
  -X POST "https://zenodo.org/api/deposit/depositions?access_token=$TOKEN" \
  -d @/tmp/deposit.json)
ID=$(echo "$DEP"    | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')
BUCKET=$(echo "$DEP" | python3 -c 'import json,sys; print(json.load(sys.stdin)["links"]["bucket"])')

# 2. upload the file
curl -s -X PUT --upload-file "$TARBALL" \
  "$BUCKET/$(basename "$TARBALL")?access_token=$TOKEN" > /dev/null

# 3. inspect the draft in the browser before committing to it
echo "https://zenodo.org/uploads/$ID"

# 4. only when it looks right:
# curl -s -X POST "https://zenodo.org/api/deposit/depositions/$ID/actions/publish?access_token=$TOKEN"
```

Try the whole thing against the sandbox first — <https://sandbox.zenodo.org> behaves
identically and issues throwaway DOIs. It needs its own separate token.

Afterwards, add a related identifier by hand pointing at the repository
(`https://github.com/Rahps97/BNN-Robustness-Verification`, relation *is supplement to*),
since nothing did it for you, and add the DOE award as in §1 step 6.

---

## 3. What the prepared files do, and their limits

| File | Read by | Notes |
| --- | --- | --- |
| `.zenodo.json` | Zenodo, on GitHub-release archiving | **Takes precedence over `CITATION.cff`** — if both are present Zenodo ignores the CFF entirely. Keep them in sync by hand. |
| `CITATION.cff` | GitHub's *"Cite this repository"* panel | Also consumed by citation managers and by GitHub's APA/BibTeX export. |

Deliberate omissions from `.zenodo.json`:

- **`version`** — Zenodo falls back to the git tag name, so the record's version can
  never drift from the release. Do not hardcode it.
- **`publication_date`** — Zenodo uses the release date.
- **`doi`** — leaving it out is what tells Zenodo to mint a fresh one.
- **DOE award DE-SC0021323** — not in Zenodo's award vocabulary; see §1 step 6.

One thing that is *not* an omission and looks redundant: `related_identifiers` contains
the GitHub repository URL as well as the arXiv DOI. Zenodo normally adds the repository
link itself, but any `related_identifiers` supplied in `.zenodo.json` **replaces** the
automatic one wholesale rather than merging with it, so it has to be restated. If you
prefer the link to pin the exact release, change it to
`https://github.com/Rahps97/BNN-Robustness-Verification/tree/v1.0.0` before cutting the
release.

ORCIDs: two of the three are asserted.

- Zheng Zhang — `0000-0002-2292-0030`, confirmed against his ORCID employment record at
  UCSB ECE.
- Seyran Saeedi — `0000-0003-1646-3870`, **confirmed by the author**. The registry record
  lists a Virginia Commonwealth University affiliation, which is a previous position; the
  paper lists her as independent with the work performed at UCSB, and the affiliation
  strings in `.zenodo.json` and `CITATION.cff` say so. An ORCID iD identifies the person,
  not the affiliation, so the mismatch is expected and is not an error.
- Rahul Singh — no ORCID could be found; the creator entry deliberately carries none.

Adding one later is a metadata edit and does not change the DOI.

Field shapes differ between the two files and both are required as written:
`.zenodo.json` takes the **bare** iD (`0000-0003-1646-3870`) — Zenodo's legacy
deserializer passes it through `idutils.normalize_orcid`, which strips a URL prefix
anyway, but bare is the form the rest of the file already uses. `CITATION.cff` takes the
**full URL** (`https://orcid.org/0000-0003-1646-3870`), which CFF 1.2.0 enforces by
regex. Both files were re-validated after the change: `cffconvert --validate` reports
CFF 1.2.0 valid, and `.zenodo.json` passes a structural check against the legacy
deserializer's expectations (allowed top-level keys, `is_orcid` on every creator,
`detect_identifier_schemes` on every related identifier, `funder::award` grant split).

---

## 4. Which version to archive

**Archive the merged `main` (released as `v1.0.0`), not `paper-results-v1`.**

The two candidates:

- **`paper-results-v1`** (annotated tag at commit `47b877e`) is the exact code state
  that produced the published numbers.
- **current `main`** additionally: fixes a Python 3.9 crash that aborted *every*
  satisfiable Table V query; removes two misclassification encodings that could emit
  false robustness certificates; fixes a hardcoded absolute path and a stale
  perturbation bound; adds CPU fallbacks; and adds `verify_paper.py`, which re-checks
  every reported number in Tables III to VII in about half a minute with no GPU, no
  solver licence, no network and no hardware.

The reason to prefer `main` is that Reviewer 1's objection was precisely that the
evaluation could not be reproduced from the repository. Minting a permanent identifier
on `47b877e` would answer that objection with an artefact that *crashes on the reviewer's
Python* before producing a single Table V result, and that still contains encodings
capable of certifying a network as robust when it is not. A DOI is forever and is cited
in the reference list; it should not point at code that cannot run.

The usual argument for archiving the tag — "the DOI must carry the bits that made the
numbers" — does not bite here, because the later commits **change no reported number**.
The repository documents this explicitly (`README.md`, and the commit that added the
tag): the post-tag FEM changes alter solver behaviour but no published value, and
`verify_paper.py` at `main` reproduces the tables. So archiving `main` gives you the
published numbers *and* code that runs.

**The trade-off, stated plainly.** `main` is not bit-for-bit the code that was executed
to produce the paper's tables. Someone auditing at the level of "which exact bytes ran"
gets that from `paper-results-v1`, not from the DOI. Two things mitigate it, and one of
them needs an action from you:

1. `paper-results-v1` is an ancestor of `main` and is documented in the README, together
   with the two settings (`FEM_PARAM_FLOOR_RATIO=0` and a shallow copy in the FEM worker
   dispatch) that recover the pre-fix behaviour from any later commit. But note that
   **GitHub's release ZIP contains no git history**, so the tag is not recoverable from
   inside a §1 archive — only from the repository, or from the §2 tarball, which does
   include `.git`.
2. *Optional, recommended if it is cheap:* after `v1.0.0` is published, use Zenodo's
   *New version* on the same record to also archive `paper-results-v1`, described as
   "code state that produced the published results". Both then sit under one concept
   DOI. Cite the `v1.0.0` version DOI in the paper regardless. If this is any trouble,
   skip it — it is a nicety, not a requirement.

---

## 5. What to do with the DOI once it exists

Zenodo gives you two DOIs (§1 step 5). Use them differently:

- **version DOI** → the manuscript. It pins the exact snapshot a reader will get.
- **concept DOI** → `CITATION.cff` and the README badge, so they keep working across
  future releases.

### 5.1 Manuscript, Data Availability section

Replace the `[DOI TO BE ADDED]` placeholder. Suggested wording, keeping the existing
sentence structure:

> The MNIST dataset used to train the BNNs in this work is publicly available from the
> MNIST website [ref]. All code required to construct the BNN robustness verification
> QUBOs and reproduce the experiments is available at
> `https://github.com/Rahps97/BNN-Robustness-Verification` and is permanently archived
> on Zenodo at `https://doi.org/10.5281/zenodo.XXXXXXX` [cite the new reference]. The
> software-based experiments can be rerun using the provided scripts, while reproducing
> the Fujitsu Digital Annealer and D-Wave experiments requires access to the
> corresponding hardware services.

Worth adding, given that Reviewer 1's objection was reproducibility: a sentence noting
that `python verify_paper.py` re-checks every reported number and exits non-zero on any
mismatch.

### 5.2 Manuscript, reference list

Springer Nature asks for the code to be **cited**, not merely linked, so the DOI needs
a reference-list entry as well. For `references.bib` (the bibliography style is
`ieeetr`):

```bibtex
@misc{BNNRobustnessVerificationCode,
  author    = {Singh, Rahul and Saeedi, Seyran and Zhang, Zheng},
  title     = {{BNN-Robustness-Verification}: code and data for
               ``Robustness Verification of Binary Neural Networks:
               An Ising and Quantum-Inspired Framework''},
  year      = {2026},
  publisher = {Zenodo},
  version   = {v1.0.0},
  doi       = {10.5281/zenodo.XXXXXXX},
  url       = {https://doi.org/10.5281/zenodo.XXXXXXX},
  note      = {Software}
}
```

Cite it from the Data Availability statement, and from the methods/experimental section
at the first mention of the implementation.

### 5.3 This repository

1. Uncomment the `identifiers:` block in `CITATION.cff` and put the **concept** DOI in
   it.
2. Add the badge to the top of `README.md`:

   ```markdown
   [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
   ```

3. Both changes are metadata-only; they can go into a follow-up commit after the release
   without re-cutting it.
