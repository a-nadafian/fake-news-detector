# 🚀 Deployment Guide

This guide covers various deployment options for the Fake News Detector project.

## 🌐 GitHub Deployment

### 1. Initial Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/fake-news-detector.git
cd fake-news-detector

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running the Streamlit App

```bash
# Start the Streamlit app
streamlit run app.py

# The app will open at http://localhost:8501
```

### 3. GitHub Pages (Optional)

For static documentation:
1. Go to repository Settings > Pages
2. Select source branch (usually `main`)
3. Select folder (usually `/docs`)
4. Save and wait for deployment

## ☁️ Cloud Deployment

### Streamlit Cloud

1. **Connect Repository**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Select your repository

2. **Configure App**:
   - Set main file path: `app.py`
   - Set Python version: `3.11`
   - Add requirements.txt path

3. **Deploy**:
   - Click "Deploy app"
   - Wait for build completion
   - Your app will be live at a Streamlit Cloud URL

### Heroku

1. **Create Heroku App**:
   ```bash
   heroku create your-app-name
   ```

2. **Add Buildpacks**:
   ```bash
   heroku buildpacks:add heroku/python
   ```

3. **Deploy**:
   ```bash
   git push heroku main
   ```

### Google Cloud Platform

1. **Enable App Engine**:
   ```bash
   gcloud app create
   ```

2. **Deploy**:
   ```bash
   gcloud app deploy
   ```

## 🐳 Docker Deployment

### 1. Create Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 2. Build and Run

```bash
# Build image
docker build -t fake-news-detector .

# Run container
docker run -p 8501:8501 fake-news-detector
```

## 🔧 Environment Configuration

### Environment Variables

Create a `.env` file (not committed to git):

```env
# Model paths
MODEL_PATH=models/fake_news_detector/final_model

# API keys (if using external services)
HUGGINGFACE_TOKEN=your_token_here

# App settings
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### Configuration Files

The app automatically detects configuration from:
- `models/fake_news_detector/final_model/model_config.json`
- Environment variables
- Command line arguments

## 📊 Monitoring and Logging

### Streamlit Built-in

- Access logs via Streamlit Cloud dashboard
- Monitor app performance and errors
- View user analytics

### Custom Logging

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Log important events
logger.info("Model loaded successfully")
logger.error("Error occurred during prediction")
```

## 🔒 Security Considerations

### Production Deployment

1. **HTTPS**: Always use HTTPS in production
2. **Authentication**: Consider adding user authentication
3. **Rate Limiting**: Implement API rate limiting
4. **Input Validation**: Validate all user inputs
5. **Model Security**: Protect model files from unauthorized access

### Environment Variables

- Never commit sensitive information to git
- Use environment variables for configuration
- Rotate API keys regularly

## 🚨 Troubleshooting

### Common Issues

1. **Model Not Found**:
   - Check model path in configuration
   - Ensure model files are properly uploaded
   - Verify file permissions

2. **Dependencies Issues**:
   - Update pip: `pip install --upgrade pip`
   - Clear cache: `pip cache purge`
   - Reinstall: `pip install -r requirements.txt --force-reinstall`

3. **Port Already in Use**:
   ```bash
   # Find process using port 8501
   lsof -i :8501
   
   # Kill process
   kill -9 <PID>
   ```

### Performance Optimization

1. **Model Caching**: Use Streamlit's caching for model loading
2. **Batch Processing**: Process multiple texts together when possible
3. **Resource Limits**: Monitor memory and CPU usage
4. **CDN**: Use CDN for static assets

## 📈 Scaling

### Horizontal Scaling

- Deploy multiple instances behind a load balancer
- Use container orchestration (Kubernetes, Docker Swarm)
- Implement session management for user state

### Vertical Scaling

- Increase server resources (CPU, RAM)
- Use more powerful GPUs for inference
- Optimize model architecture

## 🔄 Continuous Deployment

### GitHub Actions

The project includes GitHub Actions for:
- Automated testing
- Code quality checks
- Automatic deployment to staging/production

### Manual Deployment

```bash
# Pull latest changes
git pull origin main

# Install/update dependencies
pip install -r requirements.txt

# Restart application
# (Depends on your deployment method)
```

## 📞 Support

For deployment issues:
1. Check the troubleshooting section
2. Review GitHub Issues
3. Create a new issue with detailed information
4. Contact the maintainers

---

**Note**: Always test your deployment in a staging environment before going to production. 