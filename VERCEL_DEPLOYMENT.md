# Deploying to Vercel

This guide will help you deploy the Hybrid Cloud Orchestrator to Vercel.

## Prerequisites

1. A [Vercel account](https://vercel.com/signup) (free tier is sufficient)
2. [Vercel CLI](https://vercel.com/docs/cli) installed (optional, for CLI deployment)
3. Git repository (GitHub, GitLab, or Bitbucket) - recommended for easiest deployment

## Deployment Methods

### Method 1: Deploy via Vercel Dashboard (Recommended)

1. **Push your code to Git**:
   ```bash
   git add .
   git commit -m "Prepare for Vercel deployment"
   git push origin main
   ```

2. **Import Project on Vercel**:
   - Go to [vercel.com/new](https://vercel.com/new)
   - Click "Import Git Repository"
   - Select your repository
   - Vercel will auto-detect the configuration

3. **Configure Project**:
   - **Framework Preset**: Other (or leave as auto-detected)
   - **Root Directory**: `./` (project root)
   - **Build Command**: Leave empty (not needed for Python)
   - **Output Directory**: Leave empty
   - **Install Command**: `pip install -r requirements.txt`

4. **Deploy**:
   - Click "Deploy"
   - Wait for the build to complete
   - Your app will be live at `https://your-project-name.vercel.app`

### Method 2: Deploy via Vercel CLI

1. **Install Vercel CLI** (if not already installed):
   ```bash
   npm install -g vercel
   ```

2. **Login to Vercel**:
   ```bash
   vercel login
   ```

3. **Deploy**:
   ```bash
   vercel
   ```
   
   Follow the prompts:
   - Set up and deploy? **Yes**
   - Which scope? Select your account
   - Link to existing project? **No** (for first deployment)
   - Project name? Enter a name or press Enter for default
   - Directory? Press Enter (uses current directory)
   - Override settings? **No**

4. **Production Deployment**:
   ```bash
   vercel --prod
   ```

## Project Structure for Vercel

The project has been configured with:

- `vercel.json` - Vercel configuration file
- `api/index.py` - Serverless function wrapper
- `requirements.txt` - Python dependencies (root level)
- `.vercelignore` - Files to exclude from deployment

## Important Notes

### Static Files
- Static files (CSS, JS) are served from `/frontend/static/`
- Templates are in `/frontend/templates/`
- Vercel will automatically serve static files

### Environment Variables
If you need to set environment variables:
1. Go to your project settings on Vercel
2. Navigate to "Environment Variables"
3. Add any required variables

### Function Timeout
- Vercel's free tier has a 10-second timeout for serverless functions
- Hobby tier: 10 seconds
- Pro tier: 60 seconds
- Enterprise: Custom limits

If your `/predict` or `/deploy` endpoints take longer, consider:
- Optimizing the code
- Upgrading your Vercel plan
- Using Vercel's Edge Functions for faster responses

### File Size Limits
- Serverless function size: 50MB (free tier), 250MB (Pro)
- Request/response size: 4.5MB

### Cold Starts
- First request after inactivity may be slower (cold start)
- Subsequent requests will be faster
- Consider using Vercel's Edge Functions for better performance

## Troubleshooting

### Build Errors

**Error: "Module not found"**
- Ensure `requirements.txt` is in the root directory
- Check that all dependencies are listed

**Error: "Import error"**
- Verify the `api/index.py` file exists
- Check that paths in `backend/app.py` are correct

### Runtime Errors

**Error: "Template not found"**
- Ensure `frontend/templates/` directory is included
- Check that `.vercelignore` doesn't exclude templates

**Error: "Static file not found"**
- Verify static files are in `frontend/static/`
- Check `vercel.json` routing configuration

### Performance Issues

**Slow response times**
- Check function logs in Vercel dashboard
- Consider optimizing heavy operations
- Use caching where possible

## Testing Your Deployment

After deployment, test these endpoints:

1. **Homepage**: `https://your-project.vercel.app/`
2. **Benchmark**: `https://your-project.vercel.app/benchmark`
3. **API - Predict**: 
   ```bash
   curl -X POST https://your-project.vercel.app/predict \
     -H "Content-Type: application/json" \
     -d '{"cpu_cores":1.5,"memory_mb":1024,"latency_sensitive":0,"execution_time":120,"data_size_mb":300}'
   ```
4. **API - Benchmark Data**: `https://your-project.vercel.app/benchmark-data`

## Updating Your Deployment

### Via Git (Recommended)
1. Make changes locally
2. Commit and push to Git
3. Vercel will automatically redeploy

### Via CLI
```bash
vercel --prod
```

## Custom Domain

To use a custom domain:
1. Go to your project settings on Vercel
2. Navigate to "Domains"
3. Add your custom domain
4. Follow DNS configuration instructions

## Monitoring

- View logs: Vercel Dashboard → Your Project → Logs
- View analytics: Vercel Dashboard → Your Project → Analytics
- View function metrics: Vercel Dashboard → Your Project → Functions

## Cost Considerations

- **Free Tier**: 
  - 100GB bandwidth/month
  - 100 serverless function executions/day
  - Perfect for development and small projects

- **Pro Tier** ($20/month):
  - Unlimited bandwidth
  - 1000 serverless function executions/day
  - Better performance and support

## Additional Resources

- [Vercel Python Documentation](https://vercel.com/docs/runtimes/python)
- [Vercel Flask Examples](https://github.com/vercel/examples/tree/main/python/flask)
- [Vercel Support](https://vercel.com/support)

---

**Note**: This deployment configuration works for most Flask applications. If you encounter issues specific to your use case, refer to Vercel's documentation or support.

