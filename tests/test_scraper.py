"""
Tests for the Minecraft Wiki Scraper

Run with: pytest tests/test_scraper.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from scraper.wiki_scraper import MinecraftWikiScraper, ScraperConfig, PageMetadata


class TestScraperConfig:
    """Tests for ScraperConfig dataclass."""
    
    def test_default_config(self):
        config = ScraperConfig()
        assert config.base_url == "https://minecraft.wiki"
        assert config.requests_per_second == 1.0
        assert "NS_0" in config.allowed_namespaces
    
    def test_custom_config(self):
        config = ScraperConfig(
            requests_per_second=0.5,
            max_retries=5,
        )
        assert config.requests_per_second == 0.5
        assert config.max_retries == 5


class TestURLFiltering:
    """Tests for URL filtering logic."""
    
    @pytest.fixture
    def scraper(self, tmp_path):
        config = ScraperConfig(
            output_dir=tmp_path / "html",
            metadata_file=tmp_path / "metadata.json",
        )
        with patch.object(MinecraftWikiScraper, '_init_robot_parser'):
            scraper = MinecraftWikiScraper(config)
            # Mock robot parser to allow all URLs
            scraper.robot_parser = MagicMock()
            scraper.robot_parser.can_fetch.return_value = True
        return scraper
    
    def test_allows_valid_wiki_page(self, scraper):
        assert scraper._is_url_allowed("https://minecraft.wiki/w/Diamond")
        assert scraper._is_url_allowed("https://minecraft.wiki/w/Crafting")
    
    def test_blocks_user_pages(self, scraper):
        assert not scraper._is_url_allowed("https://minecraft.wiki/w/User:Someone")
    
    def test_blocks_special_pages(self, scraper):
        assert not scraper._is_url_allowed("https://minecraft.wiki/w/Special:Search")
    
    def test_blocks_file_pages(self, scraper):
        assert not scraper._is_url_allowed("https://minecraft.wiki/w/File:Diamond.png")
    
    def test_blocks_action_urls(self, scraper):
        assert not scraper._is_url_allowed("https://minecraft.wiki/w/Diamond?action=edit")
    
    def test_blocks_api_urls(self, scraper):
        assert not scraper._is_url_allowed("https://minecraft.wiki/api.php")


class TestSitemapParsing:
    """Tests for sitemap parsing."""
    
    @pytest.fixture
    def scraper(self, tmp_path):
        config = ScraperConfig(
            output_dir=tmp_path / "html",
            metadata_file=tmp_path / "metadata.json",
        )
        with patch.object(MinecraftWikiScraper, '_init_robot_parser'):
            scraper = MinecraftWikiScraper(config)
            scraper.robot_parser = MagicMock()
            scraper.robot_parser.can_fetch.return_value = True
        return scraper
    
    def test_parse_sitemap_xml(self, scraper):
        """Test parsing a mock sitemap response."""
        mock_sitemap = b"""<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <url><loc>https://minecraft.wiki/w/Diamond</loc></url>
            <url><loc>https://minecraft.wiki/w/Gold</loc></url>
            <url><loc>https://minecraft.wiki/w/Iron</loc></url>
        </urlset>
        """
        
        mock_response = MagicMock()
        mock_response.content = mock_sitemap
        
        with patch.object(scraper, '_fetch_with_retry', return_value=mock_response):
            urls = list(scraper._parse_sitemap("https://example.com/sitemap.xml"))
        
        assert len(urls) == 3
        assert "https://minecraft.wiki/w/Diamond" in urls


class TestPageScraping:
    """Tests for page scraping functionality."""
    
    @pytest.fixture
    def scraper(self, tmp_path):
        config = ScraperConfig(
            output_dir=tmp_path / "html",
            metadata_file=tmp_path / "metadata.json",
        )
        with patch.object(MinecraftWikiScraper, '_init_robot_parser'):
            scraper = MinecraftWikiScraper(config)
            scraper.robot_parser = MagicMock()
            scraper.robot_parser.can_fetch.return_value = True
        return scraper
    
    def test_scrape_page_extracts_content(self, scraper):
        """Test that page scraping extracts title and content."""
        mock_html = b"""
        <html>
        <head><title>Diamond - Minecraft Wiki</title></head>
        <body>
            <h1 id="firstHeading">Diamond</h1>
            <div id="mw-content-text">
                <p>Diamond is a rare mineral found deep underground.</p>
                <p>It can be used to craft powerful tools and armor.</p>
            </div>
            <a href="/w/Category:Minerals">Minerals</a>
        </body>
        </html>
        """
        
        mock_response = MagicMock()
        mock_response.content = mock_html
        mock_response.headers = {"Last-Modified": "2026-01-01"}
        
        with patch.object(scraper, '_fetch_with_retry', return_value=mock_response):
            metadata = scraper.scrape_page("https://minecraft.wiki/w/Diamond")
        
        assert metadata is not None
        assert metadata.title == "Diamond"
        assert metadata.word_count > 0
        assert "Minerals" in metadata.categories
    
    def test_skips_already_scraped_urls(self, scraper):
        """Test that already scraped URLs are skipped."""
        scraper.scraped_urls.add("https://minecraft.wiki/w/Diamond")
        
        result = scraper.scrape_page("https://minecraft.wiki/w/Diamond")
        assert result is None
    
    def test_skips_disallowed_urls(self, scraper):
        """Test that disallowed URLs are skipped."""
        result = scraper.scrape_page("https://minecraft.wiki/w/User:Test")
        assert result is None


class TestMetadataSaving:
    """Tests for metadata saving."""
    
    def test_save_metadata_creates_file(self, tmp_path):
        config = ScraperConfig(
            output_dir=tmp_path / "html",
            metadata_file=tmp_path / "metadata.json",
        )
        
        with patch.object(MinecraftWikiScraper, '_init_robot_parser'):
            scraper = MinecraftWikiScraper(config)
        
        # Add some test metadata
        scraper.metadata.append(PageMetadata(
            url="https://minecraft.wiki/w/Diamond",
            title="Diamond",
            last_modified="2026-01-01",
            scraped_at="2026-01-02T00:00:00",
            content_hash="abc123",
            word_count=100,
            categories=["Minerals"],
            file_path=str(tmp_path / "html" / "Diamond.html"),
        ))
        
        scraper.save_metadata()
        
        assert config.metadata_file.exists()


# Integration test - actually hits the network (skipped by default)
@pytest.mark.skip(reason="Integration test - requires network access")
class TestIntegration:
    """Integration tests that hit the actual wiki."""
    
    def test_fetch_single_page(self, tmp_path):
        """Test fetching a single real page."""
        config = ScraperConfig(
            output_dir=tmp_path / "html",
            metadata_file=tmp_path / "metadata.json",
            requests_per_second=0.5,  # Be extra careful
        )
        
        scraper = MinecraftWikiScraper(config)
        metadata = scraper.scrape_page("https://minecraft.wiki/w/Diamond")
        
        assert metadata is not None
        assert metadata.title == "Diamond"
        assert (tmp_path / "html" / "Diamond.html").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
