Every commit has a certain set of valid entrypoints, and a subset of those
entrypoints that *were* used in the creation of a certain set of artifacts.
The description names the unique output folder where output artifacts from this
commit are located.

? Fixing problems with old runs (e.g. they crashed because the servers all went down or they crashed because the commit had a bug.)
Ideally the outputs are the same (up to GPU noise) each time we run a commit.
A bug could be a typo in the Params or we chose a too-busy GPU queue and want to resubmit on something more available. So we cancel the in-flight and queue'd jobs, change the code and resubmit.
If we have multiple concurrent experiments they should be siblings, s.t. the iterative process of getting an experiment right (which involves changing code) can proceed independently for each branch.
But then we merge them together quickly once all are done. Avoid long-running indepenedent branches.

We pull results (pull script is also commit-specific but doesn't change often so we almost always avoid pulling per-experiment).
Analysis scripts are per-experiment, and they change often.

We can run code locally or on janelia's clusters using LSF.
We pull artifacts from the cluster filesystem into a local mirror. (rsync with --delete).
Local analysis (plots, tables, html summaries) should live where?
In the unique outdir folder? Then we have to avoid --delete and let artifacts accumulate.
IDEA: we could always have a --delete option that clears a specific subdir if we've generated trash.

Q: What if we accidentally rm or change an old experiment dir?

Q: What if we want to update an old experiment with a new analysis?
- It's possible to absorb changes backwards into old repo states.
- Add analysis as short branch to the side of old repo state.

Q: What if we want to share our results to make them reproducible?
Q: What if we want to add any other out-of-band artifacts (screenshots? AI generated summaries?)

Q: Where do our summaries go?
Q: Can figure generation reference previous experiments? What happens if we later fix a bug in them?
Q: Can runs reference previous experiments? I guess sure? So this means we keep the outdir in the commit
name as we continue to perform analyses and add artifacts?

IDEA: We could always generate a customized outdir browser based viewer for looking at images and other results
if napari / Preview doesn't cut it.

Q: Should we log all the commands that we've issued in each repo?
  They already appear in the local bash history.
  Sometimes we're issuing commands on a remote machine! (Reduce this with jrun.sh).
  Should we log them? In a structured format?
  Should we have a convention like "each commit has a single entrypoint runall()"?



Goals:
1. flexibility -- it should be easy to make changes deep in model and training code and evaluate them.
2. simplicity -- a very small amount of intuitive, readable, direct, transparent code. not too abstract. not too many names, files, functions, etc.
3. reproducibility -- the artifacts can be reproduced easily. i know the code that built each artifact and can find it quickly and read it and understand it. Most artifacts should have a very concise repro.
    The commands we actually execute on the command line are a notorious place for critical repro data to get lost.
4. (maybe?) The repros are stored in the REPO. not outside it. Rebuilding the full artifact set can be done by *replaying the repo*.
