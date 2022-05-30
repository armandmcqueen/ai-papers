import subprocess

import requests
import os
from PyPDF2 import PdfFileReader
from pathlib import Path
from PIL import Image
import fitz


org_map = {
    "deepmind": "DeepMind",
    "openai": "OpenAI",
    "openai.com": "OpenAI",
    "google": "Google",
    "google.com": "Google",
    "johns hopkins": "Johns Hopkins",
    "jhu": "Johns Hopkins",
    "google brain": "Google Brain",
    "google research": "Google Research",
    "university of toronto": "University of Toronto",
    "toronto.edu": "University of Toronto",
    "nvidia": "NVIDIA",
    "nvidia.com": "NVIDIA",
    "stanford": "Stanford",
    "microsoft": "Microsoft",
    "microsoft research": "Microsoft Research",
    "cs.washington.edu": "University of Washington",
    "university of washington": "University of Washington",
    "paul g. allen school of computer science": "Paul G. Allen School of Computer Science",
    "paul g allen school of computer science": "Paul G. Allen School of Computer Science",
    "facebook": "Facebook",
    "facebook ai": "Facebook",
    "fb.com": "Facebook",
    "huazhong university": "Huazhong University",
    "tencent": "Tencent",
    "illinois.edu": "University of Illinois at Urbana-Champaign",
    "university of illinois at urbana-champaign": "University of Illinois at Urbana-Champaign",
    "cuhk.edu.hk": "The Chinese University of Hong Kong",
    "the chinese university of hong kong": "The Chinese University of Hong Kong",
    " mit ": "MIT",
    "massachusetts institute of technology": "MIT",
    "@mit.edu": "MIT",
    "google research, brain": "Google Brain",
    "google research, blueshift": "Google BlueShift",
    "uc berkeley": "UC Berkeley",
    "university of california, berkeley": "UC Berkeley",
    "university college london": "University College London",
    "mcgill": "McGill University",
    "microsoft cloud + ai": "Microsoft Cloud + AI",
    "lmu munich": "LMU Munich",
    "sulzer": "Sulzer",
}
orgs = list(org_map.keys())
exclude_from_no_whitespace_search = {"mit"}  # MIT is a common substring in text. Too many false positives


def pdf_url(paper_id):
    return f"https://arxiv.org/pdf/{paper_id}"


def get_tmp_pdf_path(paper_id) -> Path:
    return Path(f"/tmp/{paper_id}.pdf")

def get_tmp_image_path(paper_id) -> Path:
    return Path(f"/tmp/{paper_id}.png")

def clean_cache(paper_id):
    tmp_file_path = get_tmp_pdf_path(paper_id)
    if tmp_file_path.exists():
        print(f"Removing cache file {tmp_file_path}")
        os.remove(tmp_file_path)
    tmp_image_path = get_tmp_image_path(paper_id)
    if tmp_image_path.exists():
        print(f"Removing cache file {tmp_image_path}")
        os.remove(tmp_image_path)

def remove_whitespace(text):
    return ''.join([char for char in text if not char.isspace()])

def extract_affiliations(paper_id, display_front_page=True):

    tmp_file_path = get_tmp_pdf_path(paper_id)
    if not tmp_file_path.exists():
        print(f"Cache file {tmp_file_path} does not exist. Downloading...")
        pdf_content = requests.get(pdf_url(paper_id)).content
        with open(tmp_file_path, 'wb') as f:
            f.write(pdf_content)
        print("Download complete.")

    with open(tmp_file_path, 'rb') as f:
        reader = PdfFileReader(f)
        first_page = reader.getPage(0)
        txt = first_page.extractText()


    matches = set()
    # Rough matching

    print(txt)
    # Find exact (case-insensitive) string matches in the text
    for org in orgs:
        if org in txt.lower():
            matches.add(org)

    # Find matches when whitespace is removed. PDF text can have odd whitespaces
    # such as "Universit y C ollege L ondon"
    txt_no_whitespace = remove_whitespace(txt).lower()
    for org in orgs:
        search_term = remove_whitespace(org).lower()
        if search_term in exclude_from_no_whitespace_search:
            continue
        if search_term in txt_no_whitespace:
            matches.add(org)


    # Compress multiple representations of the same org
    deduped_orgs = set()
    for match in matches:
        deduped_orgs.add(org_map[match])

    for org in deduped_orgs:
        print(org)


    if display_front_page:
        subprocess.run(["open", tmp_file_path])
        # doc = fitz.open(tmp_file_path)
        # page = doc.load_page(0)  # number of page
        # pix = page.get_pixmap()
        # tmp_image_path = get_tmp_image_path(paper_id)
        # pix.save(tmp_image_path)
        #
        # img = Image.open(tmp_image_path)
        # img.show()
    return deduped_orgs



if __name__ == '__main__':
    chinchilla_paper_id = "2203.15556"
    clip_paper_id = "2103.00020"
    scaling_laws_paper_id = "2001.08361"
    transformers_paper_id = "1706.03762"
    dalle_paper_id = "2102.12092"
    retro_paper_id = "2112.04426"
    lamda_paper_id = "2201.08239"
    megatron_paper_id = "2104.04473"
    gopher_paper_id = "2112.11446"
    roberta_paper_id = "1907.11692"
    deepspeed_megatron_paper_id = "2201.11990"
    transformer_scale_efficiently_paper_id = "2109.10686"

    paper_id = transformers_paper_id

    extract_affiliations(paper_id, display_front_page=True)

    # clean_cache(paper_id)

