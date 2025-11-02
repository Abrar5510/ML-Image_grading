# Deployment Guide - ML Image Grading System

This guide covers deploying the ML Image Grading web application to Render and other cloud platforms.

## Quick Deploy to Render

### Option 1: One-Click Deploy (Recommended)

1. **Fork this repository** to your GitHub account

2. **Sign up for Render** at [render.com](https://render.com)

3. **Create New Web Service**:
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Render will auto-detect the `render.yaml` configuration

4. **Configure Environment Variables** (Optional):
   - `MODEL_PATH`: Path to pre-trained model (leave empty to use heuristics)
   - `MAX_WORKERS`: Number of Gunicorn workers (default: 4)

5. **Deploy**: Click "Create Web Service"
   - Render will build and deploy automatically
   - Initial deployment takes ~10-15 minutes
   - Your app will be available at `https://your-app-name.onrender.com`

### Option 2: Manual Deploy with Docker

1. **Install Render CLI**:
```bash
brew install render  # macOS
# or
npm install -g render-cli  # npm
```

2. **Login to Render**:
```bash
render login
```

3. **Deploy**:
```bash
render deploy
```

## Deploy to Other Platforms

### Deploy to Heroku

1. **Install Heroku CLI**:
```bash
brew tap heroku/brew && brew install heroku  # macOS
```

2. **Login and Create App**:
```bash
heroku login
heroku create ml-image-grading
```

3. **Set Stack to Container**:
```bash
heroku stack:set container
```

4. **Deploy**:
```bash
git push heroku main
```

5. **Scale**:
```bash
heroku ps:scale web=1
```

### Deploy to Google Cloud Run

1. **Install Google Cloud SDK**:
```bash
brew install google-cloud-sdk  # macOS
```

2. **Login and Set Project**:
```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

3. **Build and Deploy**:
```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/ml-image-grading
gcloud run deploy ml-image-grading \
  --image gcr.io/YOUR_PROJECT_ID/ml-image-grading \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --timeout 300
```

### Deploy to AWS ECS (Fargate)

1. **Build and Push to ECR**:
```bash
aws ecr create-repository --repository-name ml-image-grading
docker build -t ml-image-grading .
docker tag ml-image-grading:latest ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/ml-image-grading:latest
aws ecr get-login-password --region REGION | docker login --username AWS --password-stdin ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com
docker push ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/ml-image-grading:latest
```

2. **Create ECS Task and Service** via AWS Console or CLI

### Deploy to DigitalOcean App Platform

1. **Connect GitHub Repository** in DigitalOcean dashboard

2. **Configure App**:
   - Type: Web Service
   - Dockerfile: `./Dockerfile`
   - HTTP Port: 10000
   - Instance Size: Basic (1 GB RAM minimum)

3. **Deploy**: Click "Create Resources"

## Local Development with Docker

### Using Docker Compose (Recommended)

```bash
# Build and start
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

Access at `http://localhost:5000`

### Using Docker Directly

```bash
# Build image
docker build -t ml-image-grading .

# Run container
docker run -p 5000:10000 \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  -e DEBUG=true \
  ml-image-grading
```

## Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `PORT` | Server port | 10000 | No |
| `DEBUG` | Debug mode | false | No |
| `MAX_WORKERS` | Gunicorn workers | 4 | No |
| `MODEL_PATH` | Path to trained model | None | No |
| `MAX_CONTENT_LENGTH` | Max upload size (bytes) | 52428800 (50MB) | No |

## Performance Optimization

### Render Configuration

**Recommended Plan**: Starter or higher
- RAM: 512 MB minimum, 2 GB recommended
- CPU: 0.5 vCPU minimum, 1 vCPU recommended

### Scaling Considerations

1. **Horizontal Scaling**: Multiple instances behind load balancer
2. **Vertical Scaling**: Larger instance size for ML inference
3. **Persistent Storage**: Use Render disks or S3 for uploads/outputs
4. **CDN**: CloudFlare or similar for static assets

### Memory Management

For limited memory environments, reduce workers:

```bash
MAX_WORKERS=2
```

Or use a worker timeout:

```bash
gunicorn --timeout 600 app:app
```

## Monitoring and Logging

### Health Check Endpoint

```
GET /api/health
```

Returns:
```json
{
  "status": "healthy",
  "timestamp": "2024-11-02T...",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### View Logs on Render

```bash
# Via web dashboard
Dashboard → Your Service → Logs

# Via CLI
render logs ml-image-grading
```

### Custom Monitoring

Integrate with services like:
- **Sentry**: Error tracking
- **DataDog**: APM and monitoring
- **New Relic**: Performance monitoring

## Troubleshooting

### Build Failures

**Issue**: Out of memory during build
```bash
# Solution: Use smaller TensorFlow build
pip install tensorflow-cpu  # Instead of tensorflow
```

**Issue**: Raw image library not found
```bash
# Ensure Dockerfile has libraw-dev
RUN apt-get install -y libraw-dev
```

### Runtime Issues

**Issue**: Model not loading
```bash
# Check model path and permissions
ls -la /app/models/
```

**Issue**: Out of memory at runtime
```bash
# Reduce workers or increase instance size
MAX_WORKERS=2
```

### Upload Issues

**Issue**: File too large
```bash
# Increase max upload size in app.py
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
```

## Security Considerations

1. **HTTPS**: Always use HTTPS in production (Render provides this automatically)
2. **File Validation**: Only allowed file types are processed
3. **Size Limits**: Max upload size enforced
4. **Rate Limiting**: Consider adding rate limiting for API endpoints
5. **Authentication**: Add authentication for production use

### Adding Basic Auth (Optional)

Create `.env` file:
```
BASIC_AUTH_USERNAME=admin
BASIC_AUTH_PASSWORD=secure_password
```

Add to `app.py`:
```python
from flask_httpauth import HTTPBasicAuth

auth = HTTPBasicAuth()

@auth.verify_password
def verify_password(username, password):
    return username == os.getenv('BASIC_AUTH_USERNAME') and \
           password == os.getenv('BASIC_AUTH_PASSWORD')

@app.route('/')
@auth.login_required
def index():
    return render_template('index.html')
```

## Backup and Data Persistence

### Backup Models and Data

```bash
# On Render, enable disk backups in dashboard
# Or backup to S3
aws s3 sync /app/models s3://your-bucket/models/
aws s3 sync /app/data s3://your-bucket/data/
```

### Restore

```bash
aws s3 sync s3://your-bucket/models/ /app/models/
aws s3 sync s3://your-bucket/data/ /app/data/
```

## CI/CD Integration

### GitHub Actions

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Render

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Deploy to Render
        run: |
          curl -X POST ${{ secrets.RENDER_DEPLOY_HOOK_URL }}
```

## Cost Estimation

### Render Pricing (as of 2024)

- **Starter**: $7/month (512 MB RAM, 0.5 CPU)
- **Standard**: $25/month (2 GB RAM, 1 CPU) - **Recommended**
- **Pro**: $85/month (4 GB RAM, 2 CPU)

Additional costs:
- **Disk Storage**: $0.25/GB/month
- **Bandwidth**: Free (100 GB/month included)

## Support

- **Documentation**: [render.com/docs](https://render.com/docs)
- **Community**: [community.render.com](https://community.render.com)
- **Status**: [status.render.com](https://status.render.com)

## Next Steps

1. Deploy to Render
2. Upload a pre-trained model (optional)
3. Configure custom domain (optional)
4. Set up monitoring
5. Enable automatic deployments

---

For more information, visit the [main README](README.md).
