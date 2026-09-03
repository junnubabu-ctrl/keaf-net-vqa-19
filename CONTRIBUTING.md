# Contributing

Use a focused branch and pull request. Every change that affects a research result must include a
test, configuration update, and claim-ledger impact assessment. Do not commit data, model weights,
generated indexes, credentials, private paths, or manually edited result tables.

Before opening a pull request, run:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m compileall -q src tests
PYTHONPATH=src python -m evitrust_vqa.cli smoke --fixture data/fixtures/tiny_fixture.json
```

