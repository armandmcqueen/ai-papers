from pathlib import Path
import click

from utils.arxiv_query import get_paper_info_by_id
from utils.markdown import MarkdownDoc, MarkdownTitle, MarkdownText, MarkdownLink
from utils.markdown import MarkdownNewline as Newline
from utils.arxiv_entry import ArxivEntry


def section_break(md):
    md.add(Newline())
    md.add(Newline())
    md.add(Newline())


def empty_line(md):
    md.add(Newline())
    md.add(Newline())


def generate_notes_page(paper_info: ArxivEntry, page_path: Path):

    md = MarkdownDoc()
    md.add(MarkdownTitle(paper_info.title, level=1, link=paper_info.abstract_url))
    section_break(md)

    md.add(MarkdownLink("PDF", paper_info.pdf_url))
    empty_line(md)
    md.add(MarkdownText(f"Published {paper_info.published_as_month_and_year}"))

    section_break(md)
    md.add(MarkdownTitle("Notes", level=2))
    empty_line(md)
    md.add(MarkdownText("Notes go here..."))

    section_break(md)
    md.add(MarkdownTitle("Paper Details", level=2))
    empty_line(md)
    md.add(MarkdownText(f"Authors: {', '.join(paper_info.authors)}"))
    empty_line(md)
    if paper_info.comment:
        md.add(MarkdownText(f"Arxiv Comments: {paper_info.comment}"))
        empty_line(md)
    md.add(MarkdownText(paper_info.abstract))

    with open(page_path, "w") as f:
        md.write(f)


# Add Notes to index
def add_readme_entre(paper_info: ArxivEntry, notes_page_path: Path, notes_page: Path):
    md = MarkdownDoc()

    md.add(Newline())
    md.add(MarkdownText("- "))
    md.add(MarkdownLink(paper_info.title, notes_page_path))
    md.add(MarkdownText(f". {paper_info.published_as_month_and_year}."))

    with open(notes_page, "a") as f:
        md.write(f)


# Add paper to reading list
def add_reading_list_entry(paper_info: ArxivEntry, reading_list_file: Path):
    md = MarkdownDoc()

    md.add(Newline())
    md.add(MarkdownText("- "))
    md.add(MarkdownLink(paper_info.title, paper_info.pdf_url))
    md.add(MarkdownText(f". {paper_info.published_as_month_and_year}"))
    with open(reading_list_file, "a") as f:
        md.write(f)


def slugify(name):
    return name.lower().replace(" ", "_")


def get_notes_page_path(paper_info: ArxivEntry, name: str) -> Path:
    if name is not None:
        markdown_page_name = f"notes/{slugify(name)}.md"
    else:
        markdown_page_name = f"notes/{paper_info.paper_id}.md"
    return Path(markdown_page_name)


@click.command()
@click.option('--paper', help='Arxiv ID of the paper to generate, e.g. 1909.08053')
@click.option('--name', help='Short name for the paper, used for file names. E.g. megatron_lm_1')
@click.option('--overwrite', is_flag=True, help='Overwrite existing notes page if already exists')
@click.option('--notes', is_flag=True, help='Generate notes page and adds to README instead of just adding to reading list')
def add_paper(paper, name=None, overwrite=False, notes=False):

    index_file = "README.md" if notes else "reading_list.md"
    index_file = Path(index_file)
    paper_info = get_paper_info_by_id(paper)[0]
    notes_page_path = get_notes_page_path(paper_info, name)

    msg = f"Adding '{paper_info.title}' to {index_file.name}."
    if notes:
        msg += f" Adding notes page to {notes_page_path.name}."
    print(msg)

    if not notes:
        # Just add the paper to the list of papers, don't generate a notes page
        add_reading_list_entry(paper_info, index_file)
    else:
        # Generate a notes page and add it to the README
        if not overwrite and notes_page_path.exists():
            raise FileExistsError(f"{notes_page_path} already exists, not overwriting")

        generate_notes_page(paper_info, notes_page_path)
        add_readme_entre(paper_info, notes_page_path, index_file)


if __name__ == '__main__':
    add_paper()


# if __name__ == '__main__':
#     # paper_id = '1803.01097'
#     paper_id = "1706.03762"
#     # paper_info = get_paper_info_by_id(paper_id)
#     # print(paper_info[0])
#     paper_info = get_paper_info_by_id(paper_id)[0]  # type: ArxivEntry
#
#     generate_markdown_page(paper_info)
#     generate_markdown_link(paper_info)