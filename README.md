# Research Papers Management System

A comprehensive research papers management system with multiple implementations: **FastAPI** and **Node.js** versions.

## Features

- 📚 **Paper Management**: Add, edit, delete, and organize research papers
- 🔍 **arXiv Integration**: Search and import papers directly from arXiv
- 📊 **Analytics Dashboard**: Visualize reading progress and paper statistics
- 🏷️ **Topic Categorization**: Organize papers by topics and research areas
- 📈 **Reading Progress Tracking**: Track papers as "read", "reading", or "want to read"
- 🔎 **Advanced Filtering**: Filter by status, year, authors, and topics
- 📱 **Responsive Design**: Works seamlessly on desktop and mobile devices

## Technology Stacks

### FastAPI Version

- **Backend**: FastAPI with Python 3.8+
- **Data**: JSON file storage with pandas processing
- **Charts**: Plotly.js for interactive visualizations
- **Frontend**: Modern HTML5, CSS3, and vanilla JavaScript
- **API Docs**: Automatic OpenAPI/Swagger documentation

### Node.js Version

- **Backend**: Express.js with modern ES6+
- **Data**: JSON file storage with native JavaScript processing
- **Validation**: Joi for data validation
- **Charts**: Plotly.js for interactive visualizations
- **Frontend**: Modern HTML5, CSS3, and vanilla JavaScript

## Quick Start

### FastAPI Version

1. **Install Dependencies**

   ```bash
   cd fastapi-papers
   pip install fastapi uvicorn pandas plotly requests
   ```

2. **Run the Application**

   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

3. **Access the Application**
   - Main App: http://localhost:8000
   - API Docs: http://localhost:8000/docs

### Node.js Version

1. **Install Dependencies**

   ```bash
   cd nodejs-papers
   npm install
   ```

2. **Run the Application**

   ```bash
   npm start
   ```

3. **Access the Application**
   - Main App: http://localhost:3000
   - API endpoints available at /api/\*

## Project Structure

```
Papers-main/
├── Papers.py                 # Original Streamlit application
├── paper_with_topics.json    # Sample data file
├── fastapi-papers/           # FastAPI implementation
│   ├── main.py              # FastAPI application
│   ├── models.py            # Pydantic data models
│   ├── database.py          # Data operations
│   ├── templates/
│   │   └── index.html       # Frontend template
│   └── static/
│       ├── styles.css       # Application styles
│       └── app.js           # Frontend JavaScript
└── nodejs-papers/           # Node.js implementation
    ├── package.json         # Node.js dependencies
    ├── server.js            # Express application
    ├── services/            # Business logic
    │   ├── database.js      # Data operations
    │   ├── arxiv.js         # arXiv API integration
    │   └── validation.js    # Data validation schemas
    ├── routes/              # API routes
    │   ├── papers.js        # Paper CRUD operations
    │   ├── arxiv.js         # arXiv search endpoints
    │   └── analytics.js     # Analytics endpoints
    └── public/              # Static files
        ├── index.html       # Frontend template
        ├── styles.css       # Application styles
        └── app.js           # Frontend JavaScript
```

## API Endpoints

Both implementations provide the same REST API:

### Papers

- `GET /api/papers` - List all papers
- `POST /api/papers` - Add a new paper
- `PUT /api/papers/{id}` - Update a paper
- `DELETE /api/papers/{id}` - Delete a paper

### arXiv Integration

- `GET /api/arxiv/search` - Search arXiv papers
  - Query params: `query`, `max_results`

### Analytics

- `GET /api/analytics` - Get analytics data including charts and insights

## Data Format

Papers are stored in JSON format with the following structure:

```json
{
  "id": "unique-id",
  "title": "Paper Title",
  "authors": "Author Name(s)",
  "year": 2024,
  "venue": "Conference/Journal",
  "url": "https://paper-url.com",
  "pdf_url": "https://pdf-url.com",
  "description": "Paper abstract or description",
  "topics": ["machine learning", "computer vision"],
  "status": "read|reading|want to read",
  "date_added": "2024-01-01T00:00:00Z"
}
```

## Features in Detail

### Paper Management

- **Add Papers**: Manual entry with complete metadata
- **Edit Papers**: Update any field including status and topics
- **Delete Papers**: Remove papers with confirmation
- **Status Tracking**: Three status levels for reading progress

### arXiv Integration

- **Search**: Query arXiv database with custom terms
- **Import**: One-click import from search results
- **Metadata**: Automatic extraction of title, authors, abstract, and categories

### Analytics Dashboard

- **Statistics**: Total papers, reading progress breakdown
- **Charts**: Interactive visualizations using Plotly.js
  - Status distribution (pie chart)
  - Papers by year (bar chart)
  - Top topics (horizontal bar chart)
- **Insights**: Automated analysis of reading patterns

### Filtering and Search

- **Status Filter**: Filter by reading status
- **Year Filter**: Filter by publication year
- **Text Search**: Search across titles, authors, and topics
- **Combined Filters**: Multiple filters work together
- **Real-time**: Instant filtering as you type

## Keyboard Shortcuts

- `Ctrl/Cmd + N`: Add new paper
- `Ctrl/Cmd + F`: Focus search filter
- `Ctrl/Cmd + R`: Reset all filters
- `Escape`: Close modal dialogs

## Browser Support

- Chrome 60+
- Firefox 60+
- Safari 12+
- Edge 79+

## Development

### FastAPI Development

```bash
# Install development dependencies
pip install black pytest

# Run with auto-reload
uvicorn main:app --reload

# Format code
black *.py

# Run tests
pytest
```

### Node.js Development

```bash
# Install development dependencies
npm install --save-dev nodemon

# Run with auto-reload
npm run dev

# Run tests
npm test
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with modern web technologies
- arXiv API for research paper data
- Plotly.js for interactive charts
- Inspired by the need for better research paper organization

## Support

For questions or issues:

1. Check the API documentation at `/docs` (FastAPI) or review the code
2. Look through existing issues
3. Create a new issue with detailed information

---

**Choose Your Stack**: Both FastAPI and Node.js versions provide identical functionality. Choose based on your team's expertise and infrastructure preferences.
