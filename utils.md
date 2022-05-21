# Utilities

## Arxiv Utility

### Add paper to reading list
To add paper to reading list
```
python add_paper.py --paper 1909.08053
```
Does not handle deduplication.

### Test automatic affiliation extraction

```
python add_paper.py --paper 1909.08054 --dryrun
```

### Add paper notes

To add paper to the README and create notes page
```
python add_paper.py --paper 1909.08053 --name megatron_lm_1 --notes
```
`--name` param is used for the file name of the paper's notes page.
