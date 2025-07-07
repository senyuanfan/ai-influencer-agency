from fastmcp import FastMCP
import pandas as pd
import pathlib
import os
import requests
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

# -----------------------------------------------------------------------------
# FastMCP server instance
# -----------------------------------------------------------------------------

mcp = FastMCP(
    name="Tools Server",
    instructions=(
        "Expose lightweight tools that enable LLMs and other MCP clients to "
        "inspect local CSV files via pandas, browse the web, search for content, "
        "and navigate websites to gather information.\n\n"
        "Web browsing features include DuckDuckGo search, link extraction, "
        "content filtering, and site navigation with customizable keywords.\n\n"
        "Set the environment variable MCP_FILE_ROOTS to one or more directory "
        "paths (comma‑separated) that the server is allowed to read."
    )
)

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _resolve_path(filename: str) -> pathlib.Path:
    """Construct the full, absolute path for a filename based on MCP_FILE_ROOTS.

    Args:
        filename: The base name of the file (e.g., 'tweets.csv').

    Returns:
        A resolved pathlib.Path object.

    Raises:
        NotADirectoryError: If the resolved root path is not a directory.
    """
    mcp_roots_str = os.getenv("MCP_FILE_ROOTS", "data")
    print(f"MCP_FILE_ROOTS: {mcp_roots_str}")
    # Use the first directory specified in MCP_FILE_ROOTS
    root_dir_str = mcp_roots_str.split(",")[0].strip()
    root_dir = pathlib.Path(root_dir_str).expanduser().resolve()

    if not root_dir.is_dir():
        raise NotADirectoryError(f"MCP_FILE_ROOTS path is not a valid directory: {root_dir}")

    full_path = root_dir / filename
    print(f"Resolved path: {full_path}")
    return full_path


def _load_df(path: str) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame with standardized processing.

    This function performs the following steps:
      - Resolves the absolute file path using `_resolve_path`.
      - Reads the CSV into a DataFrame.
      - Resets the DataFrame index.

    Args:
        path: The filename or relative path of the CSV file within the root directory.

    Returns:
        A pandas DataFrame with processed and sorted data.
    """
    df = pd.read_csv(_resolve_path(path))

    return df


# -----------------------------------------------------------------------------
# Web Browsing Tools
# -----------------------------------------------------------------------------

import re
from markdownify import markdownify
from requests.exceptions import RequestException

@mcp.tool()
def search_web(query: str, num_results: int = 10, keywords: list[str] = None) -> str:
    """Perform a web search using DuckDuckGo and return results as markdown.

    Args:
        query: The search query string.
        num_results: Maximum number of results to return (default: 10).
        keywords: Optional list of keywords to filter results. If provided, only results 
                 containing at least one keyword will be included.
                 Default: ['LLM', 'Agentic AI', 'Machine Learning', 'Artificial Intelligence', 
                          'Neural Networks', 'Deep Learning', 'ChatGPT', 'GPT', 'Claude']

    Returns:
        Search results formatted as markdown with titles, URLs, and snippets.
    """
    try:
        from duckduckgo_search import DDGS
        
        # Default AI-related keywords if none provided
        if keywords is None:
            keywords = [
                'LLM', 'Agentic AI', 'Machine Learning', 'Artificial Intelligence',
                'Neural Networks', 'Deep Learning', 'ChatGPT', 'GPT', 'Claude',
                'AI Models', 'Foundation Models', 'Generative AI'
            ]
        
        # Perform search
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num_results))
        
        if not results:
            return f"No results found for query: {query}"
        
        # Filter by keywords if provided
        if keywords:
            filtered_results = []
            for result in results:
                content_to_search = f"{result.get('title', '')} {result.get('body', '')}".lower()
                if any(keyword.lower() in content_to_search for keyword in keywords):
                    filtered_results.append(result)
            results = filtered_results
        
        if not results:
            return f"No results found matching keywords {keywords} for query: {query}"
        
        # Format results as markdown
        markdown_output = f"# Search Results for: {query}\n\n"
        for i, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            url = result.get('href', 'No URL')
            snippet = result.get('body', 'No description')
            
            markdown_output += f"## {i}. {title}\n"
            markdown_output += f"**URL:** {url}\n"
            markdown_output += f"**Description:** {snippet}\n\n"
        
        return markdown_output
        
    except ImportError:
        return "Error: duckduckgo-search package not installed. Run: pip install duckduckgo-search"
    except Exception as e:
        return f"Error performing search: {str(e)}"


@mcp.tool()
def extract_links(url: str, filter_keywords: list[str] = None, link_text_only: bool = False) -> str:
    """Extract all links from a webpage, optionally filtered by keywords.

    Args:
        url: The URL of the webpage to extract links from.
        filter_keywords: Optional list of keywords to filter links. Only links whose text 
                        or URL contains at least one keyword will be included.
                        Default: ['LLM', 'Agentic AI', 'Machine Learning', 'AI', 'Artificial Intelligence']
        link_text_only: If True, only return the link text and URLs. If False, include surrounding context.

    Returns:
        Links formatted as markdown, or an error message if the request fails.
    """
    try:
        from bs4 import BeautifulSoup
        
        # Default AI-related keywords if none provided
        if filter_keywords is None:
            filter_keywords = [
                'LLM', 'Agentic AI', 'Machine Learning', 'AI', 'Artificial Intelligence',
                'Neural Networks', 'Deep Learning', 'ChatGPT', 'GPT', 'Claude'
            ]
        
        # Fetch the webpage
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        links = soup.find_all('a', href=True)
        
        if not links:
            return f"No links found on {url}"
        
        # Filter links by keywords if provided
        filtered_links = []
        for link in links:
            link_text = link.get_text(strip=True)
            link_url = link['href']
            
            if filter_keywords:
                search_content = f"{link_text} {link_url}".lower()
                if any(keyword.lower() in search_content for keyword in filter_keywords):
                    filtered_links.append((link_text, link_url))
            else:
                filtered_links.append((link_text, link_url))
        
        if not filtered_links:
            return f"No links found matching keywords {filter_keywords} on {url}"
        
        # Format as markdown
        markdown_output = f"# Links from: {url}\n\n"
        if filter_keywords:
            markdown_output += f"**Filtered by keywords:** {', '.join(filter_keywords)}\n\n"
        
        for i, (text, link_url) in enumerate(filtered_links, 1):
            # Handle relative URLs
            if link_url.startswith('/'):
                from urllib.parse import urljoin
                link_url = urljoin(url, link_url)
            elif not link_url.startswith(('http://', 'https://')):
                continue  # Skip invalid URLs
            
            if link_text_only:
                markdown_output += f"{i}. [{text}]({link_url})\n"
            else:
                markdown_output += f"## {i}. {text}\n"
                markdown_output += f"**URL:** {link_url}\n\n"
        
        return markdown_output
        
    except ImportError:
        return "Error: beautifulsoup4 package not installed. Run: pip install beautifulsoup4"
    except RequestException as e:
        return f"Error fetching the webpage: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"


@mcp.tool()
def visit_webpage(base_url: str) -> str:
    """Visits a webpage at the given URL and returns its content as a markdown string.

    Args:
        base_url: The base URL of the webpage to visit.

    Returns:
        The content of the webpage converted to Markdown, or an error message if the request fails.
    """
    try:
        # Send a GET request to the URL
        response = requests.get(base_url)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Convert the HTML content to Markdown
        markdown_content = markdownify(response.text).strip()

        # Remove multiple line breaks
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

        return markdown_content

    except RequestException as e:
        return f"Error fetching the webpage: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"
    

@mcp.tool()
def browse_site_for_content(base_url: str, keywords: list[str] = None, max_pages: int = 5) -> str:
    """Browse a website looking for pages containing specific keywords.

    Args:
        base_url: The base URL of the website to browse.
        keywords: List of keywords to search for in page content.
                 Default: ['LLM', 'Agentic AI', 'Machine Learning', 'AI', 'Artificial Intelligence']
        max_pages: Maximum number of pages to check (default: 5).

    Returns:
        A summary of relevant pages found, formatted as markdown.
    """
    try:
        from bs4 import BeautifulSoup
        from urllib.parse import urljoin, urlparse
        
        # Default AI-related keywords if none provided
        if keywords is None:
            keywords = [
                'LLM', 'Agentic AI', 'Machine Learning', 'AI', 'Artificial Intelligence',
                'Neural Networks', 'Deep Learning', 'ChatGPT', 'GPT', 'Claude'
            ]
        
        visited_urls = set()
        relevant_pages = []
        urls_to_visit = [base_url]
        
        base_domain = urlparse(base_url).netloc
        
        while urls_to_visit and len(visited_urls) < max_pages:
            current_url = urls_to_visit.pop(0)
            
            if current_url in visited_urls:
                continue
                
            try:
                response = requests.get(current_url, timeout=10)
                response.raise_for_status()
                visited_urls.add(current_url)
                
                # Parse content
                soup = BeautifulSoup(response.text, 'html.parser')
                page_text = soup.get_text().lower()
                
                # Check if page contains keywords
                matching_keywords = [kw for kw in keywords if kw.lower() in page_text]
                
                if matching_keywords:
                    title = soup.find('title')
                    title_text = title.get_text(strip=True) if title else "No title"
                    
                    # Get a snippet of relevant content
                    snippet = ""
                    for kw in matching_keywords[:2]:  # Get context for first 2 matching keywords
                        kw_lower = kw.lower()
                        idx = page_text.find(kw_lower)
                        if idx != -1:
                            start = max(0, idx - 100)
                            end = min(len(page_text), idx + 100)
                            snippet += f"...{page_text[start:end]}... "
                    
                    relevant_pages.append({
                        'url': current_url,
                        'title': title_text,
                        'keywords': matching_keywords,
                        'snippet': snippet.strip()
                    })
                
                # Find more links to explore (only from same domain)
                links = soup.find_all('a', href=True)
                for link in links[:10]:  # Limit to avoid too many URLs
                    href = link['href']
                    full_url = urljoin(current_url, href)
                    parsed_url = urlparse(full_url)
                    
                    if (parsed_url.netloc == base_domain and 
                        full_url not in visited_urls and 
                        full_url not in urls_to_visit):
                        urls_to_visit.append(full_url)
                        
            except Exception as e:
                continue  # Skip problematic URLs
        
        # Format results
        if not relevant_pages:
            return f"No pages found containing keywords {keywords} on {base_url}"
        
        markdown_output = f"# Content Found on {base_url}\n\n"
        markdown_output += f"**Search Keywords:** {', '.join(keywords)}\n"
        markdown_output += f"**Pages Checked:** {len(visited_urls)}\n"
        markdown_output += f"**Relevant Pages Found:** {len(relevant_pages)}\n\n"
        
        for i, page in enumerate(relevant_pages, 1):
            markdown_output += f"## {i}. {page['title']}\n"
            markdown_output += f"**URL:** {page['url']}\n"
            markdown_output += f"**Matching Keywords:** {', '.join(page['keywords'])}\n"
            markdown_output += f"**Context:** {page['snippet']}\n\n"
        
        return markdown_output
        
    except ImportError:
        return "Error: beautifulsoup4 package not installed. Run: pip install beautifulsoup4"
    except Exception as e:
        return f"Error browsing site: {str(e)}"

# -----------------------------------------------------------------------------
# CSV Tools 
# -----------------------------------------------------------------------------

@mcp.tool()
def get_files() -> list[str]:
    """List all files in the directory specified by MCP_FILE_ROOTS environment variable.

    Returns:
        A list of strings containing the names of all files found in the directory
        and its subdirectories.

    Example:
        get_files()
    """
    path = os.getenv("MCP_FILE_ROOTS", "data")
    return [f.name for f in pathlib.Path(path).glob("**/*") if f.is_file()]

@mcp.tool()
def load_csv(path: str) -> dict:
    """Return column names and row count of the CSV file at the specified path.

    Args:
        path: The filename or relative path of the CSV file.

    Returns:
        A dictionary containing 'columns' (list of column names) and 'rows' (integer count).

    Note: Always call this function using keyword arguments (e.g., `load_csv(path=...)`).

    Example:
        load_csv(path='tweets.csv')
    """
    df = pd.read_csv(_resolve_path(path))
    return {"columns": df.columns.tolist(), "rows": len(df)}


@mcp.tool()
def head(path: str, n: int = 5) -> str:
    """Return the first *n* rows of the CSV file as a Markdown formatted string.

    Args:
        path: The filename or relative path of the CSV file (e.g., 'tweets.csv').
        n: The number of rows to return (default is 5).

    Returns:
        A string containing the first *n* rows formatted as Markdown.

    Note: Always call this function using keyword arguments (e.g., `head(path=..., n=...)`).

    Example:
        head(path='tweets.csv', n=10)
    """
    df = pd.read_csv(_resolve_path(path))
    return df.head(n).to_markdown()


@mcp.tool()
def describe(path: str) -> str:
    """Return basic descriptive statistics for the CSV file (using pandas DataFrame.describe).

    The output includes statistics like count, mean, std deviation, min, max, and quartiles
    for numerical columns.

    Args:
        path: The filename or relative path of the CSV file (e.g., 'tweets.csv').

    Returns:
        A string containing the descriptive statistics formatted as Markdown.

    Note: Always call this function using keyword arguments (e.g., `describe(path=...)`).

    Example:
        describe(path='tweets.csv')
    """
    df = pd.read_csv(_resolve_path(path))
    return df.describe().to_markdown()


@mcp.tool()
def query(path: str, expr: str) -> str:
    """Filter rows in the CSV file using a pandas DataFrame.query expression and return the result.

    Args:
        path: The filename or relative path of the CSV file (e.g., 'tweets.csv').
        expr: A string containing the query expression (e.g., 'likes > 1000 and 转发数量 > 50').
              Refer to pandas documentation for query syntax.

    Returns:
        A string containing the matching rows formatted as Markdown, or an error message if the query fails.

    Note: Always call this function using keyword arguments (e.g., `query(path=..., expr=...)`).

    Example:
        query(path='tweets.csv', expr='likes >= 100')
    """
    df = pd.read_csv(_resolve_path(path))
    try:
        result = df.query(expr)
    except Exception as exc:
        return f"Error evaluating query: {exc}"
    return result.to_markdown()


# -----------------------------------------------------------------------------
# Markdown File Tools
# -----------------------------------------------------------------------------

@mcp.tool()
def read_markdown(path: str) -> str:
    """Read the content of a Markdown (.md) file.

    Args:
        path: The filename or relative path of the Markdown file (e.g., 'notes.md').

    Returns:
        The content of the file as a string, or an error message if the file
        is not found, not a .md file, or cannot be read.

    Note: Always call this function using keyword arguments (e.g., `read_markdown(path=...)`).

    Example:
        read_markdown(path='roadmap.md')
    """
    try:
        full_path = _resolve_path(path)
        if full_path.suffix.lower() != ".md":
            return f"Error: File is not a Markdown file (.md): {path}"
        if not full_path.is_file():
            return f"Error: Markdown file not found: {path}"
        
        content = full_path.read_text(encoding='utf-8')
        return content
    except NotADirectoryError as e:
        return f"Error resolving path: {e}"
    except Exception as e:
        return f"Error reading Markdown file {path}: {e}"

@mcp.tool()
def write_markdown(path: str, content: str) -> str:
    """Write (or overwrite) content to a Markdown (.md) file.

    WARNING: This will overwrite the existing file content if the file already exists.

    Args:
        path: The filename or relative path of the Markdown file (e.g., 'output.md').
              The file must have a .md extension.
        content: The string content to write to the file.

    Returns:
        A success message if the write operation was successful, or an error
        message if the path is invalid, not a .md file, or writing fails.

    Note: Always call this function using keyword arguments (e.g., `write_markdown(path=..., content=...)`).

    Example:
        write_markdown(path='new_roadmap.md', content='# New Plan\n- Step 1')
    """
    try:
        full_path = _resolve_path(path)
        if full_path.suffix.lower() != ".md":
            return f"Error: File must be a Markdown file (.md): {path}"

        # Ensure parent directory exists (optional, but good practice)
        # full_path.parent.mkdir(parents=True, exist_ok=True)
        
        full_path.write_text(content, encoding='utf-8')
        return f"Successfully wrote content to Markdown file: {path}"
    except NotADirectoryError as e:
        return f"Error resolving path: {e}"
    except Exception as e:
        return f"Error writing to Markdown file {path}: {e}"


# -----------------------------------------------------------------------------
# Image Generation Tools
# -----------------------------------------------------------------------------

@mcp.tool()
def generate_image(prompt: str, filename: str = None, size: str = "1024x1024") -> str:
    """Generate an image using OpenAI's DALL-E 3 model based on a text prompt.

    Args:
        prompt: The text description of the image to generate.
        filename: Optional filename for the saved image (without extension). If not provided, 
                 a timestamp-based name will be used.
        size: Image size, must be one of "1024x1024", "1792x1024", or "1024x1792" (default: "1024x1024").

    Returns:
        A string containing the path to the saved image file, or an error message if generation fails.

    Note: Always call this function using keyword arguments (e.g., `generate_image(prompt=..., filename=...)`).
    Requires OPENAI_API_KEY environment variable to be set.

    Example:
        generate_image(prompt='A serene mountain landscape at sunset', filename='mountain_sunset')
    """
    try:
        # Check if OpenAI API key is available
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "Error: OPENAI_API_KEY environment variable not set"

        # Validate size parameter
        valid_sizes = ["1024x1024", "1792x1024", "1024x1792"]
        if size not in valid_sizes:
            return f"Error: size must be one of {valid_sizes}, got '{size}'"

        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)

        # Generate image using DALL-E 3
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size,
            n=1
        )

        # Get the image URL from the response
        image_url = response.data[0].url

        # Generate filename if not provided
        if filename is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dalle_generated_{timestamp}"

        # Ensure filename has .png extension
        if not filename.endswith('.png'):
            filename += '.png'

        # Download and save the image
        img_response = requests.get(image_url)
        img_response.raise_for_status()

        # Save to MCP_FILE_ROOTS directory
        full_path = _resolve_path(filename)
        
        with open(full_path, 'wb') as f:
            f.write(img_response.content)

        return f"Successfully generated and saved image: {filename}"

    except Exception as e:
        return f"Error generating image: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")