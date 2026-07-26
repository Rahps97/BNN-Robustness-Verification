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
| **Is the licence really MIT?** | yes for the code, plus CC BY 4.0 for the data — reviewed, see [§6](#6-the-licence-review) | `LICENSE` is inside the archive; changing it later needs a whole new version |
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

---

## 6. The licence review

> **This is a documentation note, not legal advice.** It records what was checked and
> what the sources say. Three points in it are genuine legal questions rather than
> documentation ones, and they are marked **[LEGAL]**. Those should go to UCSB's Office
> of Technology & Industry Alliances or to counsel, not be settled by reading this file.

`LICENSE` was originally added during the reproducibility work as a placeholder rather
than on the authors' instruction, so §0 above told you to settle it before publishing.
It has now been reviewed. The outcome:

**Keep MIT for the code. Add CC BY 4.0 for the data. Add a `NOTICE` file.**

`LICENSE` is unchanged. `LICENSE-DATA` and `NOTICE` are new. `CITATION.cff` lists both
SPDX identifiers, `.zenodo.json` keeps `"license": "MIT"` (the legacy field takes one
value and the deposit is predominantly software) and states the split in its
description, and the README carries a *Licence and reuse* section.

### 6.1 Why MIT is right for the code

**Nothing in the dependency graph constrains it.** Every runtime dependency is
permissive: Apache-2.0 (`qubovert`, `dimod`, `dwave-samplers` and the rest of Ocean),
BSD-3-Clause (`torch`, `torchvision`, `numpy`, `scipy`, `scikit-learn`), MIT
(`z3-solver`), MPL-2.0 (`tqdm`) and matplotlib's PSF-style licence. No GPL, LGPL or
AGPL anywhere. The one weak-copyleft item, `tqdm`, is MPL-2.0, which is *file-level*
copyleft: §3.2 bites only on MPL files you modify and redistribute, and §3.3 expressly
permits combining unmodified MPL files into a larger work under other terms. `tqdm` is
used unmodified and is not redistributed here.

One historical worry was checked and is closed: D-Wave's proprietary end-user licence
does *not* cover any Ocean package on PyPI. It applies to `dwave-inspectorapp`, a
closed-source visualisation bundle distributed off-PyPI behind an opt-in
`dwave install inspector`. Nothing here installs it.

**`gurobipy` does not bear on the code licence.** Its EULA is a contract between Gurobi
and whoever installs and runs the package; it is not a copyright licence that propagates
to source which calls the API. The repository ships an `import gurobipy` and nothing of
Gurobi's, and `verify_paper.py` reproduces the Gurobi column from the recorded logs
without the package at all. Two hygiene points that do follow from it: never commit a
`gurobi.lic`, a licence key or the wheel; and note that a reviewer who simply
`pip install gurobipy`s gets the size-limited evaluation licence (~2,000 variables),
under which the 28x28 instance at 2,235 variables will not run — the README already says
this.

**`FEM.py` is not derived from anyone's released code.** This was the item most likely to
cause a problem, and it does not. The Free Energy Machine of Shen et al. (*Nature
Computational Science* **5**, 322–332, 2025, doi:10.1038/s43588-025-00782-0) does have a
released implementation, at `github.com/Fanerst/FEM`, archived at
doi:10.5281/zenodo.14874189. `FEM.py` was compared against all of `FEM/*.py` in that
repository line by line, ignoring blank lines and comments: **zero runs of three or more
matching lines**, and exactly one shared non-trivial line, `warnings.filterwarnings('ignore')`.
The two agree on the mathematics and on some option names (`rmsprop`/`adam`, `h_factor`,
the `inv`/`exp` annealing modes) because they implement the same published method;
`FEM.py` is batched over hyperparameter candidates with its own chunking and hash-based
RNG, which the reference is not.

That independence matters more than it might sound, because **`Fanerst/FEM` has no
licence file at all** — verified against the GitHub API (`license: null`, `/license`
returns 404) and the full recursive tree. A GitHub repository with no licence is
all-rights-reserved by default. The Zenodo snapshot of the same commit *is* CC BY 4.0,
but that grant lives only in the record metadata; the zip contains no licence file
either. Since nothing was copied, neither position binds this repository. Algorithms and
mathematics are not copyrightable; only expression is. The README now states the
reimplementation explicitly, which is the right thing to have on record if a reviewer
asks why the authors' code was not reused. **[LEGAL]** if any of `FEM.py` were later
found to be a transliteration rather than a reimplementation, the analysis changes; the
line-level comparison is strong evidence but it is evidence, not a legal opinion.

**Two functions genuinely are third-party, and this is the one real defect the review
found.** Both are Apache-2.0, both were unattributed or under-attributed:

| Function | Source | Copyright |
| --- | --- | --- |
| `FEM.py::beta_range` | `dwave-neal` 0.5.x, `neal/sampler.py::_default_ising_beta_range` | 2018 D-Wave Systems Inc. |
| `bnn_as_qubo.py::le_zero_constraint_to_eq_zero_constraint` | `qubovert`, `_pcbo.py::PCBO.add_constraint_le_zero` | 2020 Joseph T. Iosue |

`beta_range` is verbatim — docstring, comments and all, including upstream's "such at"
typo — apart from the function name and one constant (`log(100)` became `log(10000)`).
The `qubovert` one carries a bare "Inspired by PCBO.add_constraint_le_zero
implementation" comment but no copyright or licence notice, and the ancilla loop, the
branch structure, the two warning strings and the "don't mutate the P" comment are
carried over essentially intact.

Apache-2.0 §4(a)/(b) require the copyright notices to travel with a redistribution and
require modifications to be stated. **This does not require changing the project
licence** — Apache-2.0 is permissive and one-way compatible with MIT, and the two
functions simply remain Apache-2.0 inside an otherwise-MIT distribution. It requires
notices, which is what `NOTICE` now provides. Putting them in a top-level `NOTICE`
satisfies §4 at the distribution level; **inline headers on the two functions would be
better still**, and are a five-line change to `FEM.py` and `bnn_as_qubo.py` that was
deliberately left to the authors rather than made here.

**The venue asks for less than this.** Springer Nature's code policy requires a Code
Availability statement and a permanent identifier — *"providing a GitHub link only is not
sufficient"* — and on licensing says only that *"authors are encouraged to manage
subsequent code versions and to use a license approved by the open source initiative"*,
with no named licence. Nature Portfolio's reporting-standards page uses the same
sentence. npj Unconventional Computing's own submission guidelines require a Code
availability statement in the Methods and mention neither licences nor DOI-minting
repositories. MIT is OSI-approved, so this is already satisfied. Note also that the
article licence is a separate matter: npj UC publishes under CC BY **or** CC BY-NC-ND at
the author's election, and Springer Nature states it *"does not claim intellectual
property rights for datasets connected to published papers"*, so the repository's licence
is entirely the authors' call and need not match the article's.

**Neither funder requires anything more.** NSF's Public Access Plan 2.0 and the 2022
OSTP memo cover publications and scientific data; the OSTP memo does not contain the word
"software" at all. PAPPG Supplement 2 (NSF 26-202, effective January 2026) moved software
into the *sharing expectation* and added one advisory sentence — material *"should be
assigned permissive licenses that allow for public reuse"* — with no named licence and no
OSI requirement. MIT is exactly a permissive licence. DOE is weaker still: 2 CFR 910
Subpart D is scoped to for-profit recipients and does not apply to a university award;
the applicable provisions (GNP-119, GNP-821-US) invoke 2 CFR 200.315 and Bayh-Dole and
mention software nowhere; and the DOE Public Access Plan says outright that re-use rights
*"are not a mandatory element of DOE's Plan"*. DOE CODE, which in any case binds
contractors and labs rather than grantees, even offers "Closed Source" as a project type.
Both funders do get a standing government-use licence (2 CFR 200.315(b)/(d)) — that is a
licence to the Government, not to the public, and it neither requires nor prevents any
public licence.

### 6.2 The patent question, which is the one to actually ask someone about

**[LEGAL]** MIT is silent on patents. Apache-2.0 grants an express patent licence (§3)
with a retaliation-termination clause. Bayh-Dole and the funders' march-in rights attach
to *patentable inventions*, not to copyright, so they do not constrain the code licence —
but if the QCBO-to-QUBO encoding is or may become the subject of a UC invention
disclosure, the choice between MIT and Apache-2.0 is a rights-retention decision, not a
software-licensing one. MIT's silence is the more conservative option, since Apache-2.0
would grant downstream users a patent licence explicitly. Two of the authors are at UCSB,
and **UC's copyright policy generally leaves copyright in scholarly and software works
with the authors while patents run through the institution** — but the specific
disclosure and release requirements vary, and this note is not the place to resolve them.
**Confirm with UCSB's Office of Technology & Industry Alliances before publishing.** Do
not treat this paragraph as clearance.

### 6.3 Why the data get their own licence, and why that is the bigger issue

MIT is written for software. Its operative terms are about "the Software" and about
including the notice in "copies or substantial portions of the Software"; applied to a
directory of `.txt` coupling matrices and `.pth` checkpoints it is at best odd and at
worst ambiguous about what a "substantial portion" even is. Creative Commons says
directly that *"the only categories of works for which CC does not recommend its licenses
are computer software and hardware"* — and the converse is the received wisdom too.
Springer Nature's own licence chooser for data deposits offers exactly this pairing: CC0
and CC BY 4.0 for data, MIT and Apache-2.0 for software. *Scientific Data*'s repository
policy goes further and *requires* CC0 or CC BY for data, explicitly disallowing `-SA`
and `-NC` clauses. FAIR R1.1 and FAIR4RS R1.1 both require a clear licence without naming
one. So CC BY 4.0 for the data is the conventional, well-supported choice, and it costs
nothing: like MIT it permits commercial use, modification and redistribution, and asks
only for attribution, which citing the paper satisfies.

**The MNIST chain of title is the genuinely awkward part, and it is why `LICENSE-DATA`
is worded the way it is.** Findings:

- MNIST was never released under an explicit licence. Wayback snapshots of
  `yann.lecun.com/exdb/mnist/` from 2002, 2020 and 2024 contain no occurrence of
  "licen", "copyright", "terms", "permission" or "public domain".
- That page no longer serves the files. Since roughly mid-January 2025 it returns an
  empty directory index and the `.gz` files 404. **This is a live problem for the
  manuscript's Data Availability statement**, which should not point readers at a dead
  URL; point at a maintained mirror or at QMNIST, and link a Wayback snapshot.
- Every asserted licence for MNIST is third-party invention and they contradict each
  other. The widely-copied "CC BY-SA 3.0" traces to a single unsourced sentence on the
  PyMVPA site (present since 2011), copied into Keras's `mnist.py` docstring and from
  there into hundreds of repositories. HuggingFace's `ylecun/mnist` card says MIT.
  TensorFlow Datasets asserts no licence at all. Kaggle mirrors variously say CC0-1.0,
  DbCL-1.0 and "unknown". None traces to LeCun, Cortes or Burges.
- NIST Special Database 19, from which MNIST was built, is not safely "public domain"
  either: the Standard Reference Data Act (15 U.S.C. §290e) is an express carve-out from
  17 U.S.C. §105 letting NIST secure copyright in reference data. NIST has not asserted
  it for SD19, but the defensible statement is "no explicit licence; NIST asserts no US
  copyright and grants a royalty-free right to prepare derivative works", not a flat
  §105 claim.
- The strongest contrary data point is QMNIST (`facebookresearch/qmnist`), by Yadav and
  **Bottou** — a co-author of the original MNIST paper — which regenerates MNIST from
  SD19 and releases it under BSD. The people closest to the provenance treated the
  images as freely relicensable.

**[LEGAL]** Whether the shipped derived artefacts are copyright derivative works of MNIST
is untested. The binarized, downsampled subsets are the most exposed, since they still
reproduce the images and re-use MNIST's selection; the QUBO coupling matrices are
algorithmically derived numbers with no recognisable expression and are very unlikely to
be. *Feist* rejects sweat-of-the-brow and holds compilation copyright "thin", which cuts
in the authors' favour, and the EU sui generis database right almost certainly does not
apply (Art. 11(1) limits it to EU-based makers, and the 15-year term would have expired
regardless). No case law or authoritative commentary was found either way. This is
recorded so that nobody later mistakes the confidence level for higher than it is.

The practical hedges are all already in place: only derived artefacts are shipped, not
the original images; `LICENSE-DATA` grants only whatever rights the authors hold in those
artefacts and disclaims any grant over MNIST itself; and it never says the derived data
are "consistent with MNIST's licensing terms", because there were none to be consistent
with.

### 6.4 What was checked and found clean

- No GPL/LGPL/AGPL anywhere in the dependency graph.
- `TrainingNN.py`'s `Binarize` straight-through estimator is the standard Hubara-style
  idiom, not a copy of any identifiable repository; a code search for its distinctive
  lines returns no match.
- The Fujitsu and D-Wave artefacts in `data/hardware_results.tar.gz` are numeric solver
  outputs — bit vectors and timings — produced by the authors' own runs, not third-party
  software.

### 6.5 One unrelated thing noticed in passing

The recorded Gurobi logs in `data/gurobi_logs.tar.gz` begin with
`Set parameter LicenseID to value 2744792` and an academic-licence banner. That is not a
licensing problem and a licence ID is not a credential, but it is an identifier tied to a
named academic account that will be permanent once the DOI is minted. Scrub it or leave
it knowingly; do not discover it afterwards.
