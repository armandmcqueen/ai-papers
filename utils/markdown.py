from typing import List


class MarkdownElement:
    def write(self) -> str:
        raise NotImplementedError()


class MarkdownTitle(MarkdownElement):
    def __init__(self, title, level: int, link: str = None):
        self.level = level
        self.title = title
        self.link = link

    def write(self) -> str:
        marking = '#' * self.level
        title_str = self.title
        if self.link:
            title_str = f'[{title_str}]({self.link})'
        return f"{marking} {title_str}"


class MarkdownText(MarkdownElement):
    def __init__(self, text):
        self.text = text

    def write(self) -> str:
        return self.text


class MarkdownLink(MarkdownElement):
    def __init__(self, text, url):
        self.text = text
        self.url = url

    def write(self) -> str:
        return f"[{self.text}]({self.url})"


class MarkdownSpace(MarkdownElement):
    def write(self) -> str:
        return " "

class MarkdownNewline(MarkdownElement):
    def write(self) -> str:
        return "\n"

class MarkdownDoc:
    def __init__(self):
        self.sections = []  # type: List[MarkdownElement]

    def add(self, elem: MarkdownElement):
        self.sections.append(elem)

    def write(self, f):
        for section in self.sections:
            f.write(section.write())



