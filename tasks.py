from invoke import task
import yaml
import json
import datetime

def line_is_paper_entry(line):
    # print(line)
    line = line.strip()
    if len(line) > 0 and line.startswith("- [") and 'http' in line:
        return True
    return False





def extract_title(line):
    assert line_is_paper_entry(line), "Can only extract title from paper entry lines"
    clipped_line = line.lstrip('-').strip()
    t = ""
    cnt = 0

    entered_title = False
    for c in clipped_line:
        if c == '[':
            entered_title = True
            cnt += 1

            # Don't include the first left bracket
            if cnt == 1:
                continue
        if c == ']':
            cnt -= 1

        if entered_title:
            if cnt == 0:
                assert len(t) != 0, "Title should not ever be empty"
                return t
            t += c



def extract_url(line):
    assert line_is_paper_entry(line), "Can only extract URL from paper entry lines"
    clipped_line = line.lstrip('-').strip()
    # remove title and brackets
    clipped_line = clipped_line.replace(extract_title(line), '')
    clipped_line = clipped_line[2:]

    # Extract from parenthesis
    url = clipped_line.split(")")[0].replace("(", "")
    assert len(url) != 0, "URL should not ever be empty"
    return url


def extract_notes(line):
    assert line_is_paper_entry(line), "Can only extract notes from paper entry lines"
    line_minus_title = line.replace(extract_title(line), '')[2:]
    line_minus_title_and_url = line_minus_title.replace(extract_url(line), '')[2:]
    while line_minus_title_and_url.startswith("."):
        line_minus_title_and_url = line_minus_title_and_url[1:]

    return line_minus_title_and_url.strip()



def readme_lines():
    with open('README.md', 'r') as f:
        lines = []
        for line in f:
            lines.append(line.rstrip('\n'))
    return lines


def write_readme(c, new_lines):
    ts = datetime.datetime.now().isoformat()
    c.run("mkdir -p backup")
    c.run(f'cp README.md backup/README-{ts}.md')

    with open('README.md', 'w') as f:
        for line in new_lines:
            f.write(f'{line}\n')





def readme_paper_entries():
    lines = readme_lines()
    entries = []
    for line in lines:
        if line_is_paper_entry(line):
            title = extract_title(line)
            url = extract_url(line)
            notes = extract_notes(line)
            entries.append((title, url, notes))
    return entries



def identify_dupes(titles, verbose=False):
    dupes = []
    for i, title in enumerate(titles):
        unseen = titles[i+1:]
        if title in unseen:
            if verbose:
                print(f'Duplicate: {title}')
            dupes.append(title)
    return dupes



@task
def dedupe(c):
    # entries =
    # titles =
    # dupes = identify_dupes(titles)
    # print("------")
    # repeat_dupes = identify_dupes(dupes)
    #
    dupes = identify_dupes([entry[0] for entry in readme_paper_entries()])

    dedupe_map = {}
    for dupe in dupes:
        dedupe_map[dupe] = 0


    deduped_lines = []
    lines = readme_lines()
    for line in lines:
        if line_is_paper_entry(line):
            title = extract_title(line)
            if title in dupes:
                if dedupe_map[title] != 0:
                    continue
                dedupe_map[title] = 1

        deduped_lines.append(line)


    write_readme(c, deduped_lines)




