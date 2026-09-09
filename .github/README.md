# Automated CPU checks

workflows/checks.yml runs on pushes and pull requests. It installs CPU PyTorch,
verifies the package and runs synthetic CPU tests. Research projects verify their
existing MHD lock and actual import source. CI does not change that lock, access UKB
data, submit server jobs or count as GPU/distributed/scientific experiment acceptance.

The workflow uses read-only repository permissions. Its status and logs are available
in the repository Actions tab; enabling the workflow alone is not a passed check.
