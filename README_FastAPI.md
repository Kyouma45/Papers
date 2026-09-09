# Papers Management System - FastAPI Version

This is a FastAPI conversion of the original Streamlit papers management application. It provides a web-based interface for managing research papers with features like arXiv integration, filtering, and analytics.

## Features

- 📚 **Paper Management**: Add, edit, delete, and organize research papers
- 🔍 **arXiv Integration**: Automatically fetch paper details from arXiv URLs
- 🏷️ **Topic Organization**: Tag papers with topics and filter by them
- 📊 **Analytics Dashboard**: View reading statistics and visualizations
- 🔎 **Advanced Filtering**: Filter papers by status, topics, and text search
- 📱 **Responsive Design**: Works on desktop and mobile devices

## Installation

1. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:

   ```bash
   python main.py
   ```

   Or using uvicorn directly:

   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

3. **Access the Application**:
   Open your browser and go to: `http://localhost:8000`

## API Documentation

The FastAPI application automatically generates API documentation:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## API Endpoints

### Papers

- `GET /api/papers` - Get all papers
- `GET /api/papers/{paper_id}` - Get a specific paper
- `POST /api/papers` - Create a new paper
- `PUT /api/papers/{paper_id}` - Update a paper
- `DELETE /api/papers/{paper_id}` - Delete a paper

### arXiv Integration

- `POST /api/fetch-arxiv` - Fetch paper details from arXiv URL

### Filtering and Search

- `GET /api/filter-papers` - Filter papers with various criteria
- `GET /api/topics` - Get all unique topics

### Analytics

- `GET /api/analysis` - Get reading analytics and charts

## Data Storage

Papers are stored in `paper_with_topics.json` file in the same format as the original Streamlit application, ensuring compatibility.

## Key Improvements over Streamlit Version

1. **RESTful API**: Clean API endpoints for programmatic access
2. **Better Performance**: FastAPI is faster than Streamlit for API operations
3. **Separation of Concerns**: Clear separation between frontend and backend
4. **Modern Web Interface**: Custom HTML/CSS/JavaScript frontend
5. **API Documentation**: Automatic API documentation generation
6. **Concurrent Requests**: Better handling of multiple simultaneous users

## File Structure

```
├── main.py              # FastAPI application
├── models.py            # Pydantic data models
├── database.py          # Database operations
├── requirements.txt     # Python dependencies
├── templates/
│   └── index.html      # Main web interface
├── static/
│   ├── style.css       # CSS styles
│   └── app.js          # JavaScript functionality
└── paper_with_topics.json  # Data storage (created automatically)
```

## Development

To run in development mode with auto-reload:

```bash
uvicorn main:app --reload
```

## Production Deployment

For production deployment, consider using:

- **Gunicorn** with uvicorn workers
- **Nginx** as a reverse proxy
- **Docker** for containerization
- **Environment variables** for configuration

Example with Gunicorn:

```bash
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
```

## Browser Compatibility

The application works with modern browsers that support:

- ES6+ JavaScript features
- CSS Grid and Flexbox
- Fetch API

## Contributing

Feel free to submit issues and enhancement requests!
