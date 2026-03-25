import pytest
from bs4 import BeautifulSoup
from pathlib import Path
import json
from backend.preprocessing.text_cleaner import TextCleaner, CleanerConfig

class TestTextCleaner:
    
    @pytest.fixture
    def cleaner(self, tmp_path):
        # Point to tmp_path to avoid reading real data during unit tests
        config = CleanerConfig(
            html_dir=tmp_path / "html", 
            output_file=tmp_path / "out.json",
            scrape_metadata_file=tmp_path / "scrape.json"
        )
        return TextCleaner(config)

    def test_noise_removal(self, cleaner):
        """Verify that navigation and chrome elements are stripped."""
        html = """
        <div class="mw-parser-output">
            <div id="toc">Table of Contents</div>
            <p>Real content <span class="mw-editsection">[edit]</span></p>
            <table class="navbox"><tr><td>Nav</td></tr></table>
            <div class="printfooter">Retrieved from info</div>
        </div>
        """
        soup = BeautifulSoup(html, "lxml")
        content = soup.select_one(".mw-parser-output")
        
        # Manually run the noise removal steps (replicating process_single logic partially)
        NOISE_SELECTORS = ["div#toc", "span.mw-editsection", "table.navbox", "div.printfooter"]
        for sel in NOISE_SELECTORS:
            for tag in content.select(sel):
                tag.decompose()
        
        text = content.get_text(strip=True)
        assert "Table of Contents" not in text
        assert "[edit]" not in text
        assert "Nav" not in text
        assert "Retrieved from" not in text
        assert "Real content" in text

    def test_infobox_extraction(self, cleaner):
        """Verify infobox is extracted as dict and removed from DOM."""
        html = """
        <div class="mw-parser-output">
            <div class="infobox">
                <tr>
                    <th>Rarity</th>
                    <td>Common</td>
                </tr>
                <tr>
                    <th>Stackable</th>
                    <td>Yes (64)</td>
                </tr>
            </div>
            <p>Body text</p>
        </div>
        """
        soup = BeautifulSoup(html, "lxml")
        content = soup.select_one(".mw-parser-output")
        
        data = cleaner._extract_infobox(content)
        
        assert data["Rarity"] == "Common"
        assert data["Stackable"] == "Yes (64)"
        
        # Verify infobox is gone from DOM
        assert content.select_one(".infobox") is None
        assert "Rarity" not in content.get_text()

    def test_table_conversion(self, cleaner):
        """Verify wikitables convert to Markdown."""
        html = """
        <table class="wikitable">
            <tr>
                <th>Item</th>
                <th>Cost</th>
            </tr>
            <tr>
                <td>Apple</td>
                <td>2</td>
            </tr>
        </table>
        """
        soup = BeautifulSoup(html, "lxml")
        table = soup.select_one("table")
        
        md = cleaner._table_to_markdown(table)
        
        lines = md.split("\n")
        assert "| Item | Cost |" in lines[0]
        assert "| --- | --- |" in lines[1]
        assert "| Apple | 2 |" in lines[2]

    def test_section_parsing(self, cleaner):
        """Verify text is correctly assigned to headers with hierarchical breadcrumbs."""
        html = """
        <div class="mw-parser-output">
            <p>Intro text</p>
            <h2>Usage</h2>
            <p>Use it like this.</p>
            <h3>Crafting</h3>
            <p>Recipe here.</p>
            <h2>History</h2>
            <p>Old versions.</p>
        </div>
        """
        soup = BeautifulSoup(html, "lxml")
        content = soup.select_one(".mw-parser-output")
        
        # Mocking the _clean_text method which is used inside _process_sections_robust
        cleaner._clean_text = lambda s: s.strip()
        
        sections = cleaner._process_sections_robust(content)
        
        assert len(sections) == 4
        
        # Sect 1: Introduction (auto-assigned)
        assert sections[0]["heading"] == "Introduction"
        assert sections[0]["text"] == "Intro text"
        
        # Sect 2: Usage
        assert sections[1]["heading"] == "Usage"
        assert sections[1]["level"] == 2
        assert sections[1]["text"] == "Use it like this."
        
        # Sect 3: Crafting (H3 under Usage) - hierarchical breadcrumb
        assert sections[2]["heading"] == "Usage > Crafting"
        assert sections[2]["level"] == 3
        assert sections[2]["text"] == "Recipe here."
        
        # Sect 4: History (new H2 resets hierarchy)
        assert sections[3]["heading"] == "History"
        assert sections[3]["level"] == 2
        assert sections[3]["text"] == "Old versions."

    def test_sprite_handling(self, cleaner):
        """Verify sprite-text is kept and sprite-file is removed."""
        html = """
        <div class="mw-parser-output">
            <p>
                Craft with 
                <span class="sprite-file"><img src="stick.png"></span>
                <span class="sprite-text">Stick</span>
            </p>
        </div>
        """
        soup = BeautifulSoup(html, "lxml")
        content = soup.select_one(".mw-parser-output")
        
        # Run cleanup logic manually
        for s in content.select(".sprite-file"):
            s.decompose()
            
        text = content.get_text(" ", strip=True)
        assert "Stick" in text
        assert "stick.png" not in str(content)


class TestSelectorCollision:
    """Tests that the text cleaner handles duplicate div.mw-parser-output correctly.
    
    Some wiki pages (featured articles, CC-licensed protocol pages) have an
    mw-indicator badge wrapped in its own div.mw-parser-output BEFORE the main
    content div. The cleaner must select the main content div, not the badge.
    """

    FEATURED_ARTICLE_HTML = """
    <html><body>
    <div class="mw-indicators">
        <div class="mw-indicator" id="mw-indicator-featured">
            <div class="mw-parser-output">
                <span class="pixel-image" typeof="mw:File">
                    <a href="/w/Minecraft_Wiki:Featured_articles" title="This is a featured article.">
                        <img alt="Featured" src="/images/Invicon_Nether_Star.gif" width="32" height="32"/>
                    </a>
                </span>
            </div>
        </div>
    </div>
    <div class="mw-body-content" id="mw-content-text">
        <div class="mw-content-ltr mw-parser-output" dir="ltr" lang="en">
            <div class="infobox">
                <tr><th>Rarity</th><td>Common</td></tr>
            </div>
            <p>A poisonous potato is a food item that can poison the player.</p>
            <div class="mw-heading mw-heading2"><h2>Obtaining</h2></div>
            <p>Poisonous potatoes have a 2% chance of dropping when harvesting potatoes.</p>
            <div class="mw-heading mw-heading3"><h3>Mob loot</h3></div>
            <p>Zombies may drop poisonous potatoes when killed.</p>
            <div class="mw-heading mw-heading2"><h2>Usage</h2></div>
            <p>Eating a poisonous potato restores 2 hunger points.</p>
            <table class="wikitable">
                <tr><th>Stat</th><th>Value</th></tr>
                <tr><td>Hunger</td><td>2</td></tr>
            </table>
        </div>
    </div>
    </body></html>
    """

    @pytest.fixture
    def cleaner(self, tmp_path):
        config = CleanerConfig(
            html_dir=tmp_path / "html",
            output_file=tmp_path / "out.json",
            scrape_metadata_file=tmp_path / "scrape.json"
        )
        return TextCleaner(config)

    @pytest.fixture
    def html_file(self, tmp_path):
        """Write a featured article HTML to a temp file."""
        html_dir = tmp_path / "html"
        html_dir.mkdir(parents=True, exist_ok=True)
        path = html_dir / "Poisonous_Potato.html"
        path.write_text(self.FEATURED_ARTICLE_HTML, encoding="utf-8")
        return path

    def test_selects_main_content_not_indicator(self, cleaner, html_file):
        """process_single must extract text from the main content div, not the indicator badge."""
        result = cleaner.process_single(html_file)

        assert result is not None
        assert result["word_count"] > 0, (
            "Selector returned the indicator badge instead of main content"
        )

    def test_sections_extracted_from_featured_page(self, cleaner, html_file):
        """Featured article pages should have properly extracted sections."""
        result = cleaner.process_single(html_file)

        headings = [s["heading"] for s in result["sections"]]
        assert "Introduction" in headings
        assert "Obtaining" in headings
        assert "Usage" in headings

    def test_infobox_extracted_from_featured_page(self, cleaner, html_file):
        """Infobox should be extracted even on pages with indicator badges."""
        result = cleaner.process_single(html_file)

        assert result["infobox"].get("Rarity") == "Common"

    def test_tables_extracted_from_featured_page(self, cleaner, html_file):
        """Tables should be correctly converted on featured article pages."""
        result = cleaner.process_single(html_file)

        assert len(result["tables"]) >= 1
        assert "Hunger" in result["tables"][0]

    def test_breadcrumb_sections_on_featured_page(self, cleaner, html_file):
        """Hierarchical breadcrumbs should work on featured article pages."""
        result = cleaner.process_single(html_file)

        headings = [s["heading"] for s in result["sections"]]
        assert "Obtaining > Mob loot" in headings


class TestProcessSingleEndToEnd:
    """End-to-end tests for process_single with a standard (non-featured) page."""

    STANDARD_HTML = """
    <html><body>
    <div class="mw-body-content" id="mw-content-text">
        <div class="mw-content-ltr mw-parser-output" dir="ltr" lang="en">
            <div class="infobox">
                <tr><th>Type</th><td>Block</td></tr>
                <tr><th>Stackable</th><td>Yes (64)</td></tr>
            </div>
            <div id="toc">Table of Contents should be stripped</div>
            <p>Dirt is a block found abundantly in most biomes.</p>
            <div class="mw-heading mw-heading2"><h2>Obtaining</h2></div>
            <p>Dirt can be obtained by mining it with any tool or by hand.</p>
            <div class="mw-heading mw-heading3"><h3>Breaking</h3></div>
            <p>Dirt drops itself when mined. A shovel is the fastest tool.</p>
            <div class="mw-heading mw-heading2"><h2>Usage</h2></div>
            <p>Dirt can be used for farming and construction.</p>
            <table class="wikitable">
                <tr><th>Tool</th><th>Speed</th></tr>
                <tr><td>Shovel</td><td>Fast</td></tr>
                <tr><td>Hand</td><td>Slow</td></tr>
            </table>
            <table class="navbox"><tr><td>Navigation links</td></tr></table>
            <div class="printfooter">Retrieved from wiki</div>
        </div>
    </div>
    </body></html>
    """

    @pytest.fixture
    def cleaner(self, tmp_path):
        config = CleanerConfig(
            html_dir=tmp_path / "html",
            output_file=tmp_path / "out.json",
            scrape_metadata_file=tmp_path / "scrape.json"
        )
        return TextCleaner(config)

    @pytest.fixture
    def html_file(self, tmp_path):
        html_dir = tmp_path / "html"
        html_dir.mkdir(parents=True, exist_ok=True)
        path = html_dir / "Dirt.html"
        path.write_text(self.STANDARD_HTML, encoding="utf-8")
        return path

    def test_word_count_positive(self, cleaner, html_file):
        result = cleaner.process_single(html_file)
        assert result is not None
        assert result["word_count"] > 0

    def test_noise_stripped(self, cleaner, html_file):
        result = cleaner.process_single(html_file)
        all_text = " ".join(s["text"] for s in result["sections"])
        assert "Table of Contents" not in all_text
        assert "Navigation links" not in all_text
        assert "Retrieved from" not in all_text

    def test_infobox_extracted_and_removed(self, cleaner, html_file):
        result = cleaner.process_single(html_file)
        assert result["infobox"]["Type"] == "Block"
        assert result["infobox"]["Stackable"] == "Yes (64)"
        # Infobox text should not appear in sections
        all_text = " ".join(s["text"] for s in result["sections"])
        assert "Stackable" not in all_text

    def test_tables_converted(self, cleaner, html_file):
        result = cleaner.process_single(html_file)
        assert len(result["tables"]) >= 1
        assert "Shovel" in result["tables"][0]

    def test_sections_complete(self, cleaner, html_file):
        result = cleaner.process_single(html_file)
        headings = [s["heading"] for s in result["sections"]]
        assert "Introduction" in headings
        assert "Obtaining" in headings
        assert "Obtaining > Breaking" in headings
        assert "Usage" in headings

    def test_page_metadata(self, cleaner, html_file):
        result = cleaner.process_single(html_file)
        assert result["clean_path"] == "Dirt.html"
        assert result["last_processed"]  # Non-empty timestamp


class TestAutoRetryZeroWordPages:
    """Verify that 0-word entries are removed from skip set on re-initialization."""

    def test_zero_word_pages_excluded_from_skip_set(self, tmp_path):
        """Pages with word_count 0 should be re-processed on next run."""
        # Create a metadata.json with some 0-word entries
        output_file = tmp_path / "out.json"
        existing_data = {
            "pages": [
                {"clean_path": "Good_Page.html", "word_count": 500},
                {"clean_path": "Bad_Page.html", "word_count": 0},
                {"clean_path": "Another_Good.html", "word_count": 200},
            ],
            "processing_info": {}
        }
        with open(output_file, "w") as f:
            json.dump(existing_data, f)

        config = CleanerConfig(
            html_dir=tmp_path / "html",
            output_file=output_file,
            scrape_metadata_file=tmp_path / "scrape.json"
        )
        cleaner = TextCleaner(config)

        # Good pages should be in skip set
        assert "Good_Page.html" in cleaner.processed_files
        assert "Another_Good.html" in cleaner.processed_files
        # Zero-word page should NOT be in skip set (will be retried)
        assert "Bad_Page.html" not in cleaner.processed_files
        # Zero-word entry should be removed from output data
        paths = [p["clean_path"] for p in cleaner.output_data["pages"]]
        assert "Bad_Page.html" not in paths