import wikipediaapi

wiki = wikipediaapi.Wikipedia(
    language="en",
    user_agent="MisinformationDetection (contact: 283327@student.pwr.edu.pl)",
)


def get_wikipedia(url):
    if not url:
        return "No article found"
    title = url.split("/")[-1]
    page = wiki.page(title)
    return page.summary if page.exists() else "No article found"
