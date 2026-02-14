"""
HTML Page Endpoints
===================
All inline HTML page endpoints for the web UI.
"""

import html
from datetime import datetime
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from app.routers.advanced_analytics import get_executive_summary, get_demographics_analytics, get_zone_analytics
from app.routers.behavior import get_behavior_analytics

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def root():
    """Home dashboard with live stats."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>CCTV Analytics - Home</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #0a0a0a;
                color: #e0e0e0;
                min-height: 100vh;
            }
            .container { max-width: 1400px; margin: 0 auto; padding: 20px; }

            /* Navigation */
            nav {
                background: #111;
                border-bottom: 1px solid #222;
                padding: 0 20px;
                position: sticky;
                top: 0;
                z-index: 100;
            }
            nav .nav-inner {
                max-width: 1400px;
                margin: 0 auto;
                display: flex;
                align-items: center;
                gap: 40px;
                height: 60px;
            }
            nav .logo {
                font-size: 20px;
                font-weight: bold;
                color: #fff;
                text-decoration: none;
            }
            nav .logo span { color: #3b82f6; }
            nav .nav-links { display: flex; gap: 30px; }
            nav a {
                color: #888;
                text-decoration: none;
                font-size: 14px;
                transition: color 0.2s;
            }
            nav a:hover { color: #fff; }
            nav a.active { color: #3b82f6; }

            /* Hero */
            .hero {
                text-align: center;
                padding: 60px 20px;
                background: linear-gradient(180deg, #111 0%, #0a0a0a 100%);
                border-bottom: 1px solid #222;
            }
            .hero h1 {
                font-size: 48px;
                color: #fff;
                margin-bottom: 15px;
            }
            .hero p {
                font-size: 18px;
                color: #888;
                max-width: 600px;
                margin: 0 auto 30px;
            }

            /* Stats Grid */
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 40px 0;
            }
            .stat-card {
                background: #1a1a1a;
                border-radius: 16px;
                padding: 30px;
                border: 1px solid #333;
                text-align: center;
                transition: transform 0.2s, border-color 0.2s;
            }
            .stat-card:hover {
                transform: translateY(-2px);
                border-color: #444;
            }
            .stat-icon { font-size: 32px; margin-bottom: 15px; }
            .stat-value {
                font-size: 36px;
                font-weight: bold;
                color: #fff;
                margin-bottom: 5px;
            }
            .stat-label { color: #888; font-size: 14px; }

            /* Action Cards */
            .actions-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 20px;
                margin: 40px 0;
            }
            .action-card {
                background: #1a1a1a;
                border-radius: 16px;
                padding: 30px;
                border: 1px solid #333;
                text-decoration: none;
                color: inherit;
                transition: all 0.2s;
                display: block;
            }
            .action-card:hover {
                transform: translateY(-3px);
                border-color: #3b82f6;
                box-shadow: 0 10px 40px rgba(59, 130, 246, 0.1);
            }
            .action-card .icon {
                width: 50px;
                height: 50px;
                border-radius: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 24px;
                margin-bottom: 20px;
            }
            .action-card.upload .icon { background: rgba(59, 130, 246, 0.2); }
            .action-card.process .icon { background: rgba(168, 85, 247, 0.2); }
            .action-card.map .icon { background: rgba(34, 197, 94, 0.2); }
            .action-card.api .icon { background: rgba(245, 158, 11, 0.2); }
            .action-card h3 { color: #fff; margin-bottom: 10px; font-size: 18px; }
            .action-card p { color: #888; font-size: 14px; line-height: 1.5; }

            /* Recent Activity */
            .section { margin: 40px 0; }
            .section h2 {
                color: #fff;
                margin-bottom: 20px;
                font-size: 24px;
            }
            .activity-list {
                background: #1a1a1a;
                border-radius: 16px;
                border: 1px solid #333;
                overflow: hidden;
            }
            .activity-item {
                display: flex;
                align-items: center;
                gap: 15px;
                padding: 20px;
                border-bottom: 1px solid #333;
            }
            .activity-item:last-child { border-bottom: none; }
            .activity-icon {
                width: 40px;
                height: 40px;
                border-radius: 10px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 18px;
                background: #333;
            }
            .activity-content { flex: 1; }
            .activity-title { color: #fff; font-weight: 500; }
            .activity-meta { color: #666; font-size: 13px; margin-top: 3px; }
            .activity-stat { text-align: right; }
            .activity-stat .value { color: #fff; font-weight: bold; font-size: 18px; }
            .activity-stat .label { color: #666; font-size: 12px; }

            .empty-state {
                text-align: center;
                padding: 60px 20px;
                color: #666;
            }
            .empty-state .icon { font-size: 48px; margin-bottom: 15px; }

            /* Footer */
            footer {
                text-align: center;
                padding: 40px 20px;
                color: #666;
                font-size: 14px;
                border-top: 1px solid #222;
                margin-top: 60px;
            }
            footer a { color: #3b82f6; text-decoration: none; }
        </style>
    </head>
    <body>
        <nav>
            <div class="nav-inner">
                <a href="/" class="logo">CCTV<span>Analytics</span></a>
                <div class="nav-links">
                    <a href="/" class="active">Home</a>
                    <a href="/analytics">Analytics</a>
                    <a href="/process">Process Video</a>
                    <a href="/uploads">Batch Upload</a>
                    <a href="/map">Map</a>
                    <a href="/architecture">Architecture</a>
                    <a href="/docs">API Docs</a>
                </div>
            </div>
        </nav>

        <div class="hero">
            <h1>Video Analytics Platform</h1>
            <p>Privacy-preserving analytics for retail and hospitality. Track visitors, demographics, and behavior from CCTV footage.</p>
        </div>

        <div class="container">
            <!-- Live Stats -->
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-icon">🏢</div>
                    <div class="stat-value" id="stat-venues">-</div>
                    <div class="stat-label">Active Venues</div>
                </div>
                <div class="stat-card">
                    <div class="stat-icon">👥</div>
                    <div class="stat-value" id="stat-visitors">-</div>
                    <div class="stat-label">Total Visitors</div>
                </div>
                <div class="stat-card">
                    <div class="stat-icon">🎬</div>
                    <div class="stat-value" id="stat-videos">-</div>
                    <div class="stat-label">Videos Processed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-icon">⏱️</div>
                    <div class="stat-value" id="stat-queue">-</div>
                    <div class="stat-label">In Queue</div>
                </div>
            </div>

            <!-- Quick Actions -->
            <div class="section">
                <h2>Quick Actions</h2>
                <div class="actions-grid">
                    <a href="/uploads" class="action-card upload">
                        <div class="icon">📤</div>
                        <h3>Batch Upload</h3>
                        <p>Upload multiple videos at once. Queue them for processing and track progress in real-time.</p>
                    </a>
                    <a href="/process" class="action-card process">
                        <div class="icon">🎥</div>
                        <h3>Process Single Video</h3>
                        <p>Upload a video file or paste a YouTube URL to analyze immediately.</p>
                    </a>
                    <a href="/map" class="action-card map">
                        <div class="icon">🗺️</div>
                        <h3>View Map</h3>
                        <p>See all venues on an interactive map with visitor counts and zone data.</p>
                    </a>
                    <a href="/docs" class="action-card api">
                        <div class="icon">📡</div>
                        <h3>API Documentation</h3>
                        <p>Integrate with our REST API. Full OpenAPI spec with interactive testing.</p>
                    </a>
                </div>
            </div>

            <!-- Recent Activity -->
            <div class="section">
                <h2>Recent Activity</h2>
                <div class="activity-list" id="activity-list">
                    <div class="empty-state">
                        <div class="icon">📊</div>
                        <div>Loading recent activity...</div>
                    </div>
                </div>
            </div>
        </div>

        <footer>
            <p>CCTV Analytics Platform &bull; <a href="/docs">API Docs</a> &bull; Database: <span id="db-type">SQLite</span></p>
        </footer>

        <script>
            async function loadStats() {
                try {
                    // Load venues count
                    const venuesResp = await fetch('/venues');
                    const venuesRaw = await venuesResp.json();
                    const venues = venuesRaw.data || venuesRaw;
                    document.getElementById('stat-venues').textContent = venues.length;

                    // Load batch stats
                    const batchResp = await fetch('/api/batch/stats');
                    const batchRaw = await batchResp.json();
                    const batch = batchRaw.data || batchRaw;
                    document.getElementById('stat-videos').textContent = batch.queue.completed;
                    document.getElementById('stat-queue').textContent = batch.queue.pending + batch.queue.processing;
                    document.getElementById('stat-visitors').textContent = batch.total_visitors_detected.toLocaleString();

                } catch (e) {
                    console.error('Failed to load stats:', e);
                }
            }

            async function loadActivity() {
                try {
                    const resp = await fetch('/api/batch/jobs?limit=10');
                    const raw = await resp.json();
                    const data = raw.data || raw;

                    const list = document.getElementById('activity-list');

                    if (data.jobs.length === 0) {
                        list.innerHTML = `
                            <div class="empty-state">
                                <div class="icon">📭</div>
                                <div>No activity yet. Upload some videos to get started!</div>
                            </div>
                        `;
                        return;
                    }

                    list.innerHTML = data.jobs.map(job => {
                        const icon = {
                            'pending': '⏳',
                            'processing': '⚙️',
                            'completed': '✅',
                            'failed': '❌'
                        }[job.status] || '📹';

                        const time = job.completed_at || job.started_at || job.created_at;
                        const timeStr = time ? new Date(time).toLocaleString() : '';

                        return `
                            <div class="activity-item">
                                <div class="activity-icon">${icon}</div>
                                <div class="activity-content">
                                    <div class="activity-title">${job.video_name || 'Video'}</div>
                                    <div class="activity-meta">${job.venue_id} &bull; ${timeStr}</div>
                                </div>
                                <div class="activity-stat">
                                    <div class="value">${job.visitors_detected || '-'}</div>
                                    <div class="label">visitors</div>
                                </div>
                            </div>
                        `;
                    }).join('');
                } catch (e) {
                    console.error('Failed to load activity:', e);
                }
            }

            // Load on page load
            loadStats();
            loadActivity();

            // Refresh every 5 seconds
            setInterval(() => {
                loadStats();
                loadActivity();
            }, 5000);
        </script>
    </body>
    </html>
    """



@router.get("/analytics", response_class=HTMLResponse)
async def analytics_home():
    """Analytics home page - lists all venues with their dashboards."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Analytics - CCTV Analytics</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #0a0a0a;
                color: #e0e0e0;
                min-height: 100vh;
            }
            nav {
                background: #111;
                border-bottom: 1px solid #222;
                padding: 0 20px;
                position: sticky;
                top: 0;
                z-index: 100;
            }
            nav .nav-inner {
                max-width: 1400px;
                margin: 0 auto;
                display: flex;
                align-items: center;
                gap: 40px;
                height: 60px;
            }
            nav .logo {
                font-size: 20px;
                font-weight: bold;
                color: #fff;
                text-decoration: none;
            }
            nav .logo span { color: #3b82f6; }
            nav .nav-links { display: flex; gap: 30px; }
            nav .nav-links a {
                color: #888;
                text-decoration: none;
                font-size: 14px;
                transition: color 0.2s;
            }
            nav .nav-links a:hover { color: #fff; }
            nav .nav-links a.active { color: #3b82f6; }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 40px 20px;
            }
            h1 { color: #fff; margin-bottom: 10px; }
            .subtitle { color: #888; margin-bottom: 40px; }
            .venues-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
                gap: 20px;
            }
            .venue-card {
                background: #1a1a1a;
                border: 1px solid #333;
                border-radius: 12px;
                padding: 25px;
                transition: border-color 0.2s, transform 0.2s;
            }
            .venue-card:hover {
                border-color: #3b82f6;
                transform: translateY(-2px);
            }
            .venue-card h3 {
                color: #fff;
                margin-bottom: 8px;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .venue-card .venue-id {
                color: #666;
                font-size: 12px;
                font-family: monospace;
                background: #0a0a0a;
                padding: 2px 8px;
                border-radius: 4px;
            }
            .venue-stats {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 15px;
                margin: 20px 0;
            }
            .venue-stat {
                text-align: center;
            }
            .venue-stat-value {
                font-size: 24px;
                font-weight: bold;
                color: #3b82f6;
            }
            .venue-stat-label {
                font-size: 11px;
                color: #888;
                text-transform: uppercase;
            }
            .venue-actions {
                display: flex;
                gap: 10px;
                margin-top: 15px;
            }
            .btn {
                background: #3b82f6;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 8px;
                font-size: 14px;
                cursor: pointer;
                text-decoration: none;
                flex: 1;
                text-align: center;
            }
            .btn:hover { background: #2563eb; }
            .btn-secondary { background: #333; }
            .btn-secondary:hover { background: #444; }
            .btn-danger { background: #dc2626; }
            .btn-danger:hover { background: #b91c1c; }
            .btn-sm { padding: 6px 12px; font-size: 12px; flex: none; }
            .empty-state {
                text-align: center;
                padding: 80px 20px;
                color: #666;
            }
            .empty-state h2 { color: #888; margin-bottom: 15px; }
            .empty-state a { color: #3b82f6; }
            .loading {
                text-align: center;
                padding: 60px;
                color: #888;
            }
            .spinner {
                width: 40px;
                height: 40px;
                border: 3px solid #333;
                border-top-color: #3b82f6;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin: 0 auto 15px;
            }
            @keyframes spin { to { transform: rotate(360deg); } }
        </style>
    </head>
    <body>
        <nav>
            <div class="nav-inner">
                <a href="/" class="logo">CCTV<span>Analytics</span></a>
                <div class="nav-links">
                    <a href="/">Home</a>
                    <a href="/analytics" class="active">Analytics</a>
                    <a href="/process">Process Video</a>
                    <a href="/uploads">Batch Upload</a>
                    <a href="/map">Map</a>
                    <a href="/architecture">Architecture</a>
                    <a href="/docs">API Docs</a>
                </div>
            </div>
        </nav>

        <div class="container">
            <h1>Analytics Dashboard</h1>
            <p class="subtitle">Select a venue to view detailed analytics</p>

            <div id="loading" class="loading">
                <div class="spinner"></div>
                Loading venues...
            </div>

            <div id="venues-grid" class="venues-grid" style="display: none;"></div>

            <div id="empty-state" class="empty-state" style="display: none;">
                <h2>No Venues Yet</h2>
                <p>Create a venue and process some videos to see analytics here.</p>
                <p style="margin-top: 20px;"><a href="/process">Process a video →</a></p>
            </div>
        </div>

        <script>
            async function deleteVenue(venueId) {
                if (!confirm(`Delete venue "${venueId}" and ALL its data (events, visitors, jobs, alerts)? This cannot be undone.`)) return;
                try {
                    const res = await fetch(`/venues/${venueId}`, { method: 'DELETE' });
                    if (res.ok) {
                        loadVenues();
                    } else {
                        const err = await res.json();
                        alert('Delete failed: ' + (err.detail || 'Unknown error'));
                    }
                } catch (e) {
                    alert('Delete failed: ' + e.message);
                }
            }

            async function loadVenues() {
                try {
                    const response = await fetch('/venues');
                    const body = await response.json();
                    const venues = body.data || body;

                    document.getElementById('loading').style.display = 'none';

                    if (!venues || venues.length === 0) {
                        document.getElementById('empty-state').style.display = 'block';
                        document.getElementById('venues-grid').style.display = 'none';
                        return;
                    }

                    // Fetch stats for each venue
                    const venueCards = await Promise.all(venues.map(async (venue) => {
                        let stats = { unique_visitors: 0, avg_dwell_minutes: 0, return_rate: 0 };
                        try {
                            const statsRes = await fetch(`/analytics/${venue.id}?days=7`);
                            if (statsRes.ok) {
                                const statsBody = await statsRes.json();
                                stats = statsBody.data || statsBody;
                            }
                        } catch (e) {}

                        return `
                            <div class="venue-card" id="card-${venue.id}">
                                <h3>
                                    ${venue.name || venue.id}
                                    <span class="venue-id">${venue.id}</span>
                                </h3>
                                <div class="venue-stats">
                                    <div class="venue-stat">
                                        <div class="venue-stat-value">${(stats.unique_visitors || 0).toLocaleString()}</div>
                                        <div class="venue-stat-label">Visitors (7d)</div>
                                    </div>
                                    <div class="venue-stat">
                                        <div class="venue-stat-value">${stats.avg_dwell_minutes || 0}m</div>
                                        <div class="venue-stat-label">Avg Dwell</div>
                                    </div>
                                    <div class="venue-stat">
                                        <div class="venue-stat-value">${Math.round((stats.return_rate || 0) * 100)}%</div>
                                        <div class="venue-stat-label">Return Rate</div>
                                    </div>
                                </div>
                                <div class="venue-actions">
                                    <a href="/analytics-dashboard/${venue.id}" class="btn">View Dashboard</a>
                                    <a href="/report/${venue.id}" class="btn btn-secondary">Report</a>
                                    <button onclick="deleteVenue('${venue.id}')" class="btn btn-danger btn-sm">Delete</button>
                                </div>
                            </div>
                        `;
                    }));

                    document.getElementById('venues-grid').innerHTML = venueCards.join('');
                    document.getElementById('venues-grid').style.display = 'grid';
                    document.getElementById('empty-state').style.display = 'none';
                } catch (error) {
                    document.getElementById('loading').innerHTML = 'Error loading venues: ' + error.message;
                }
            }

            loadVenues();
        </script>
    </body>
    </html>
    """


@router.get("/process", response_class=HTMLResponse)
async def process_page():
    """Video processing page - upload or paste YouTube URL."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Process Video - CCTV Analytics</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #0a0a0a;
                color: #e0e0e0;
                min-height: 100vh;
            }
            /* Navigation */
            nav {
                background: #111;
                border-bottom: 1px solid #222;
                padding: 0 20px;
                position: sticky;
                top: 0;
                z-index: 100;
            }
            nav .nav-inner {
                max-width: 1400px;
                margin: 0 auto;
                display: flex;
                align-items: center;
                gap: 40px;
                height: 60px;
            }
            nav .logo {
                font-size: 20px;
                font-weight: bold;
                color: #fff;
                text-decoration: none;
            }
            nav .logo span { color: #3b82f6; }
            nav .nav-links { display: flex; gap: 30px; }
            nav .nav-links a {
                color: #888;
                text-decoration: none;
                font-size: 14px;
                transition: color 0.2s;
            }
            nav .nav-links a:hover { color: #fff; }
            nav .nav-links a.active { color: #3b82f6; }
            .container {
                max-width: 800px;
                margin: 0 auto;
                padding: 40px 20px;
            }
            h1 { color: #fff; margin-bottom: 10px; }
            .subtitle { color: #888; margin-bottom: 40px; }
            .card {
                background: #1a1a1a;
                border: 1px solid #333;
                border-radius: 12px;
                padding: 30px;
                margin-bottom: 20px;
            }
            .card h3 { margin-top: 0; color: #fff; }
            label { display: block; margin-bottom: 8px; color: #aaa; font-size: 14px; }
            input[type="text"], input[type="file"] {
                width: 100%;
                padding: 12px 16px;
                border: 1px solid #333;
                border-radius: 8px;
                background: #0a0a0a;
                color: #fff;
                font-size: 16px;
                margin-bottom: 20px;
            }
            input[type="text"]:focus { border-color: #0066ff; outline: none; }
            input[type="file"] { padding: 10px; }
            button {
                background: #0066ff;
                color: white;
                border: none;
                padding: 14px 28px;
                border-radius: 8px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                width: 100%;
                transition: background 0.2s;
            }
            button:hover { background: #0052cc; }
            button:disabled { background: #333; cursor: not-allowed; }
            .divider {
                text-align: center;
                margin: 30px 0;
                color: #555;
                position: relative;
            }
            .divider::before, .divider::after {
                content: '';
                position: absolute;
                top: 50%;
                width: 45%;
                height: 1px;
                background: #333;
            }
            .divider::before { left: 0; }
            .divider::after { right: 0; }
            #progress-container {
                display: none;
                margin-top: 30px;
            }
            .progress-bar {
                background: #333;
                border-radius: 8px;
                height: 20px;
                overflow: hidden;
                margin-bottom: 15px;
            }
            .progress-fill {
                background: linear-gradient(90deg, #0066ff, #00ccff);
                height: 100%;
                width: 0%;
                transition: width 0.3s;
            }
            #status-message {
                color: #888;
                font-size: 14px;
                margin-bottom: 20px;
            }
            #result {
                display: none;
                background: #0d2818;
                border: 1px solid #1e5631;
                border-radius: 8px;
                padding: 20px;
                margin-top: 20px;
            }
            #result.error {
                background: #2d1010;
                border-color: #5c1e1e;
            }
            #result h4 { margin: 0 0 10px 0; color: #4ade80; }
            #result.error h4 { color: #f87171; }
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 15px;
                margin-top: 15px;
            }
            .stat-box {
                background: #0a0a0a;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
            }
            .stat-value { font-size: 24px; font-weight: bold; color: #fff; }
            .stat-label { font-size: 12px; color: #666; margin-top: 5px; }
            .btn-secondary {
                background: #333;
                margin-top: 20px;
            }
            .btn-secondary:hover { background: #444; }
            a { color: #3b82f6; text-decoration: none; }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <nav>
            <div class="nav-inner">
                <a href="/" class="logo">CCTV<span>Analytics</span></a>
                <div class="nav-links">
                    <a href="/">Home</a>
                    <a href="/process" class="active">Process Video</a>
                    <a href="/uploads">Batch Upload</a>
                    <a href="/map">Map</a>
                    <a href="/architecture">Architecture</a>
                    <a href="/docs">API Docs</a>
                </div>
            </div>
        </nav>

        <div class="container">
        <h1>🎥 Process Video</h1>
        <p class="subtitle">Upload a video file or paste a YouTube URL to analyze visitor traffic</p>

        <div class="card">
            <h3>Option 1: YouTube URL</h3>
            <label for="youtube-url">Paste YouTube URL (bar, restaurant, retail scenes work best)</label>
            <input type="text" id="youtube-url" placeholder="https://www.youtube.com/watch?v=...">
            <button onclick="processYouTube()" id="btn-youtube">Process YouTube Video</button>
        </div>

        <div class="divider">OR</div>

        <div class="card">
            <h3>Option 2: Upload Video</h3>
            <label for="video-file">Select video file (MP4, MOV, AVI)</label>
            <input type="file" id="video-file" accept="video/*">
            <button onclick="processUpload()" id="btn-upload">Process Uploaded Video</button>
        </div>

        <div class="card">
            <h3>Venue Details</h3>
            <label for="venue-id">Venue ID</label>
            <input type="text" id="venue-id" value="demo_venue" placeholder="my_bar">

            <label for="venue-name">Venue Name</label>
            <input type="text" id="venue-name" placeholder="My Restaurant">

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                <div>
                    <label for="venue-lat">Latitude</label>
                    <input type="number" id="venue-lat" step="0.0001" placeholder="-26.2041">
                </div>
                <div>
                    <label for="venue-lng">Longitude</label>
                    <input type="number" id="venue-lng" step="0.0001" placeholder="28.0473">
                </div>
            </div>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                <div>
                    <label for="venue-city">City</label>
                    <input type="text" id="venue-city" placeholder="Johannesburg">
                </div>
                <div>
                    <label for="venue-country">Country</label>
                    <input type="text" id="venue-country" placeholder="South Africa">
                </div>
            </div>

            <label for="venue-type">Venue Type</label>
            <select id="venue-type" style="width: 100%; padding: 12px; background: #0a0a0a; color: #fff; border: 1px solid #333; border-radius: 8px; margin-bottom: 20px;">
                <option value="">Select type...</option>
                <option value="bar">Bar</option>
                <option value="restaurant">Restaurant</option>
                <option value="cafe">Cafe</option>
                <option value="retail">Retail Store</option>
                <option value="nightclub">Nightclub</option>
                <option value="hotel">Hotel/Lodge</option>
                <option value="mall">Shopping Mall</option>
                <option value="other">Other</option>
            </select>

            <div id="map-container" style="height: 200px; border-radius: 8px; margin-bottom: 15px; background: #1a1a1a; display: flex; align-items: center; justify-content: center; color: #666;">
                <div id="map" style="width: 100%; height: 100%; border-radius: 8px;"></div>
            </div>
            <p style="font-size: 12px; color: #666; margin: 0;">Click map to set location, or enter coordinates manually</p>
        </div>

        <!-- Leaflet CSS/JS for map -->
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

        <div id="progress-container" class="card">
            <h3>Processing...</h3>
            <div class="progress-bar">
                <div class="progress-fill" id="progress-fill"></div>
            </div>
            <div id="status-message">Starting...</div>
        </div>

        <div id="result">
            <h4 id="result-title">Processing Complete</h4>
            <p id="result-message"></p>
            <div class="stats-grid" id="stats-grid">
                <div class="stat-box">
                    <div class="stat-value" id="stat-frames">0</div>
                    <div class="stat-label">Frames Processed</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value" id="stat-visitors">0</div>
                    <div class="stat-label">Unique Visitors</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value" id="stat-events">0</div>
                    <div class="stat-label">Events Created</div>
                </div>
            </div>
            <button class="btn-secondary" onclick="viewDashboard()">View Dashboard &rarr;</button>
        </div>

        <script>
            let currentJobId = null;
            let statusInterval = null;
            let venueId = 'demo_venue';
            let map = null;
            let marker = null;

            // Initialize map centered on Africa
            document.addEventListener('DOMContentLoaded', function() {
                map = L.map('map').setView([-26.2041, 28.0473], 4);  // Centered on South Africa
                L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                    attribution: '&copy; OpenStreetMap contributors'
                }).addTo(map);

                // Click to set location
                map.on('click', function(e) {
                    setLocation(e.latlng.lat, e.latlng.lng);
                });

                // Update map when inputs change
                document.getElementById('venue-lat').addEventListener('change', updateMapFromInputs);
                document.getElementById('venue-lng').addEventListener('change', updateMapFromInputs);
            });

            function setLocation(lat, lng) {
                document.getElementById('venue-lat').value = lat.toFixed(6);
                document.getElementById('venue-lng').value = lng.toFixed(6);

                if (marker) {
                    marker.setLatLng([lat, lng]);
                } else {
                    marker = L.marker([lat, lng]).addTo(map);
                }
                map.setView([lat, lng], 12);
            }

            function updateMapFromInputs() {
                const lat = parseFloat(document.getElementById('venue-lat').value);
                const lng = parseFloat(document.getElementById('venue-lng').value);
                if (!isNaN(lat) && !isNaN(lng)) {
                    setLocation(lat, lng);
                }
            }

            function getVenueData() {
                return {
                    venue_id: document.getElementById('venue-id').value.trim() || 'demo_venue',
                    venue_name: document.getElementById('venue-name').value.trim(),
                    latitude: parseFloat(document.getElementById('venue-lat').value) || null,
                    longitude: parseFloat(document.getElementById('venue-lng').value) || null,
                    city: document.getElementById('venue-city').value.trim(),
                    country: document.getElementById('venue-country').value.trim(),
                    venue_type: document.getElementById('venue-type').value
                };
            }

            function setButtonsEnabled(enabled) {
                document.getElementById('btn-youtube').disabled = !enabled;
                document.getElementById('btn-upload').disabled = !enabled;
            }

            function showProgress() {
                document.getElementById('progress-container').style.display = 'block';
                document.getElementById('result').style.display = 'none';
                setButtonsEnabled(false);
            }

            function updateProgress(percent, message) {
                document.getElementById('progress-fill').style.width = percent + '%';
                document.getElementById('status-message').textContent = message;
            }

            function showResult(success, message, stats) {
                document.getElementById('progress-container').style.display = 'none';
                document.getElementById('result').style.display = 'block';
                document.getElementById('result').className = success ? '' : 'error';
                document.getElementById('result-title').textContent = success ? 'Processing Complete' : 'Error';
                document.getElementById('result-message').textContent = message;

                if (stats) {
                    document.getElementById('stats-grid').style.display = 'grid';
                    document.getElementById('stat-frames').textContent = stats.frames || 0;
                    document.getElementById('stat-visitors').textContent = stats.visitors || 0;
                    document.getElementById('stat-events').textContent = stats.events || 0;
                } else {
                    document.getElementById('stats-grid').style.display = 'none';
                }

                setButtonsEnabled(true);
            }

            function viewDashboard() {
                window.location.href = '/analytics-dashboard/' + venueId;
            }

            async function checkStatus() {
                if (!currentJobId) return;

                try {
                    const response = await fetch('/process/status/' + currentJobId);
                    const raw = await response.json();
                    const data = raw.data || raw;

                    if (data.status === 'processing') {
                        const percent = data.frames_to_process > 0
                            ? Math.round((data.current_frame / data.frames_to_process) * 100)
                            : 0;
                        updateProgress(percent, data.message);
                    } else if (data.status === 'loading_model' || data.status === 'opening_video') {
                        updateProgress(5, data.message);
                    } else if (data.status === 'generating_events') {
                        updateProgress(90, data.message);
                    } else if (data.status === 'saving_events') {
                        updateProgress(95, data.message);
                    } else if (data.status === 'completed') {
                        clearInterval(statusInterval);
                        updateProgress(100, 'Complete!');
                        setTimeout(() => {
                            showResult(true, data.message, {
                                frames: data.current_frame || data.frames_to_process,
                                visitors: data.unique_visitors,
                                events: data.total_events
                            });
                        }, 500);
                    } else if (data.status === 'error') {
                        clearInterval(statusInterval);
                        showResult(false, data.message);
                    }
                } catch (e) {
                    console.error('Status check failed:', e);
                }
            }

            async function processYouTube() {
                const url = document.getElementById('youtube-url').value.trim();
                const venueData = getVenueData();
                venueId = venueData.venue_id;

                if (!url) {
                    alert('Please enter a YouTube URL');
                    return;
                }

                showProgress();
                updateProgress(0, 'Downloading video from YouTube...');

                try {
                    const response = await fetch('/process/youtube', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            url: url,
                            venue_id: venueId,
                            venue_name: venueData.venue_name,
                            latitude: venueData.latitude,
                            longitude: venueData.longitude,
                            city: venueData.city,
                            country: venueData.country,
                            venue_type: venueData.venue_type
                        })
                    });

                    const raw = await response.json();
                    const data = raw.data || raw;

                    if (data.job_id) {
                        currentJobId = data.job_id;
                        statusInterval = setInterval(checkStatus, 1000);
                    } else {
                        showResult(false, data.detail || raw.message || 'Failed to start processing');
                    }
                } catch (e) {
                    showResult(false, 'Error: ' + e.message);
                }
            }

            async function processUpload() {
                const fileInput = document.getElementById('video-file');
                const venueData = getVenueData();
                venueId = venueData.venue_id;

                if (!fileInput.files || !fileInput.files[0]) {
                    alert('Please select a video file');
                    return;
                }

                showProgress();
                updateProgress(0, 'Uploading video...');

                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                formData.append('venue_id', venueId);

                try {
                    const response = await fetch('/process/upload', {
                        method: 'POST',
                        body: formData
                    });

                    const raw = await response.json();
                    const data = raw.data || raw;

                    if (data.job_id) {
                        currentJobId = data.job_id;
                        updateProgress(5, 'Processing started...');
                        statusInterval = setInterval(checkStatus, 1000);
                    } else {
                        showResult(false, data.detail || raw.message || 'Failed to start processing');
                    }
                } catch (e) {
                    showResult(false, 'Error: ' + e.message);
                }
            }
        </script>
        </div>
    </body>
    </html>
    """


@router.get("/dashboard/{venue_id}", response_class=HTMLResponse)
async def dashboard(venue_id: str):
    """Dashboard to view analytics for a venue."""
    venue_id = html.escape(venue_id)
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Dashboard - {venue_id} - CCTV Analytics</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * {{ box-sizing: border-box; margin: 0; padding: 0; }}
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #0a0a0a;
                color: #e0e0e0;
                min-height: 100vh;
            }}
            /* Navigation */
            nav {{
                background: #111;
                border-bottom: 1px solid #222;
                padding: 0 20px;
                position: sticky;
                top: 0;
                z-index: 100;
            }}
            nav .nav-inner {{
                max-width: 1400px;
                margin: 0 auto;
                display: flex;
                align-items: center;
                gap: 40px;
                height: 60px;
            }}
            nav .logo {{
                font-size: 20px;
                font-weight: bold;
                color: #fff;
                text-decoration: none;
            }}
            nav .logo span {{ color: #3b82f6; }}
            nav .nav-links {{ display: flex; gap: 30px; }}
            nav .nav-links a {{
                color: #888;
                text-decoration: none;
                font-size: 14px;
                transition: color 0.2s;
            }}
            nav .nav-links a:hover {{ color: #fff; }}
            nav .nav-links a.active {{ color: #3b82f6; }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                padding: 40px 20px;
            }}
            h1 {{ color: #fff; margin-bottom: 5px; }}
            .subtitle {{ color: #888; margin-bottom: 40px; }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 20px;
                margin-bottom: 40px;
            }}
            .stat-card {{
                background: #1a1a1a;
                border: 1px solid #333;
                border-radius: 12px;
                padding: 25px;
                position: relative;
            }}
            .stat-card.primary {{
                border-color: #0066ff;
                background: linear-gradient(135deg, #1a1a2e 0%, #1a1a1a 100%);
            }}
            .stat-value {{
                font-size: 42px;
                font-weight: bold;
                color: #fff;
                margin-bottom: 5px;
            }}
            .stat-card.primary .stat-value {{
                color: #4da6ff;
            }}
            .stat-label {{ color: #888; font-size: 14px; font-weight: 500; }}
            .stat-hint {{ color: #555; font-size: 11px; margin-top: 8px; }}
            .card {{
                background: #1a1a1a;
                border: 1px solid #333;
                border-radius: 12px;
                padding: 25px;
                margin-bottom: 20px;
            }}
            .card h3 {{ margin-top: 0; color: #fff; }}
            .bar-chart {{
                display: flex;
                align-items: flex-end;
                height: 200px;
                gap: 8px;
                padding-top: 20px;
            }}
            .bar {{
                flex: 1;
                background: linear-gradient(180deg, #0066ff, #0044aa);
                border-radius: 4px 4px 0 0;
                min-height: 4px;
                position: relative;
            }}
            .bar-label {{
                position: absolute;
                bottom: -25px;
                left: 50%;
                transform: translateX(-50%);
                font-size: 11px;
                color: #666;
            }}
            .pie-container {{
                display: flex;
                justify-content: space-around;
                flex-wrap: wrap;
                gap: 30px;
            }}
            .pie-section {{
                text-align: center;
            }}
            .pie-chart {{
                width: 150px;
                height: 150px;
                border-radius: 50%;
                margin: 0 auto 15px;
            }}
            .legend {{
                display: flex;
                justify-content: center;
                gap: 20px;
                flex-wrap: wrap;
            }}
            .legend-item {{
                display: flex;
                align-items: center;
                gap: 8px;
                font-size: 14px;
            }}
            .legend-color {{
                width: 12px;
                height: 12px;
                border-radius: 3px;
            }}
            a {{ color: #3b82f6; text-decoration: none; }}
            a:hover {{ text-decoration: underline; }}
            .loading {{ text-align: center; padding: 40px; color: #888; }}
            .actions {{
                display: flex;
                gap: 15px;
                margin-bottom: 30px;
            }}
            .btn {{
                background: #3b82f6;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                font-size: 14px;
                cursor: pointer;
                text-decoration: none;
            }}
            .btn:hover {{ background: #0052cc; text-decoration: none; }}
            .btn-secondary {{ background: #333; }}
            .btn-secondary:hover {{ background: #444; }}
            .section-title {{
                color: #666;
                font-size: 12px;
                text-transform: uppercase;
                letter-spacing: 1px;
                margin-bottom: 15px;
            }}
        </style>
    </head>
    <body>
        <nav>
            <div class="nav-inner">
                <a href="/" class="logo">CCTV<span>Analytics</span></a>
                <div class="nav-links">
                    <a href="/">Home</a>
                    <a href="/analytics">Analytics</a>
                    <a href="/process">Process Video</a>
                    <a href="/uploads">Batch Upload</a>
                    <a href="/map">Map</a>
                    <a href="/architecture">Architecture</a>
                    <a href="/docs">API Docs</a>
                </div>
            </div>
        </nav>

        <div class="container">
        <h1>📊 Venue Analytics</h1>
        <p class="subtitle">Venue: <strong>{venue_id}</strong> | Last 7 days</p>

        <div class="actions">
            <a href="/analytics-dashboard/{venue_id}" class="btn">Full Analytics Dashboard</a>
            <a href="/process" class="btn btn-secondary">Process New Video</a>
            <a href="/report/{venue_id}" class="btn btn-secondary">Print Report</a>
        </div>

        <div id="loading" class="loading">Loading analytics...</div>

        <div id="dashboard" style="display: none;">
            <div class="section-title">Key Metrics</div>
            <div class="stats-grid">
                <div class="stat-card primary">
                    <div class="stat-value" id="unique-visitors">-</div>
                    <div class="stat-label">People Detected</div>
                    <div class="stat-hint" id="visitor-range">Unique individuals tracked</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="confidence">-</div>
                    <div class="stat-label">Confidence</div>
                    <div class="stat-hint" id="data-quality">Track quality score</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="avg-dwell">-</div>
                    <div class="stat-label">Avg Time in View</div>
                    <div class="stat-hint">How long people stayed visible</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="peak-hour">-</div>
                    <div class="stat-label">Peak Hour</div>
                    <div class="stat-hint">Busiest time of day</div>
                </div>
            </div>

            <div class="card" id="demographics-card">
                <h3>Demographics Breakdown</h3>
                <div id="demographics-unavailable" style="text-align: center; padding: 40px; color: #666;">
                    No faces detected in video.<br>
                    Demographics require visible faces (front-facing).
                </div>
                <div class="pie-container" id="demographics-charts" style="display: none;">
                    <div class="pie-section">
                        <h4>Gender</h4>
                        <div id="gender-chart" class="pie-chart"></div>
                        <div id="gender-legend" class="legend"></div>
                    </div>
                    <div class="pie-section">
                        <h4>Age Groups</h4>
                        <div id="age-chart" class="pie-chart"></div>
                        <div id="age-legend" class="legend"></div>
                    </div>
                </div>
                </div>
            </div>

            <div class="card">
                <h3>Traffic by Hour</h3>
                <div id="hourly-chart" class="bar-chart"></div>
            </div>
        </div>

        <script>
            const venueId = '{venue_id}';

            function createPieChart(elementId, data, colors) {{
                const container = document.getElementById(elementId);
                const total = Object.values(data).reduce((a, b) => a + b, 0);

                if (total === 0) {{
                    container.style.background = '#333';
                    return;
                }}

                let gradient = 'conic-gradient(';
                let currentAngle = 0;
                const entries = Object.entries(data);

                entries.forEach(([key, value], i) => {{
                    const angle = (value / total) * 360;
                    const color = colors[i % colors.length];
                    gradient += `${{color}} ${{currentAngle}}deg ${{currentAngle + angle}}deg`;
                    if (i < entries.length - 1) gradient += ', ';
                    currentAngle += angle;
                }});

                gradient += ')';
                container.style.background = gradient;
            }}

            function createLegend(elementId, data, colors) {{
                const container = document.getElementById(elementId);
                const total = Object.values(data).reduce((a, b) => a + b, 0);

                container.innerHTML = Object.entries(data).map(([key, value], i) => {{
                    const percent = total > 0 ? Math.round((value / total) * 100) : 0;
                    return `
                        <div class="legend-item">
                            <div class="legend-color" style="background: ${{colors[i % colors.length]}}"></div>
                            <span>${{key}}: ${{percent}}%</span>
                        </div>
                    `;
                }}).join('');
            }}

            function createBarChart(elementId, data) {{
                const container = document.getElementById(elementId);
                const maxValue = Math.max(...data.map(d => d.visitors), 1);

                container.innerHTML = data.map(d => {{
                    const height = (d.visitors / maxValue) * 100;
                    return `
                        <div class="bar" style="height: ${{Math.max(height, 2)}}%">
                            <span class="bar-label">${{d.hour}}:00</span>
                        </div>
                    `;
                }}).join('');
            }}

            async function loadData() {{
                try {{
                    const [statsRes, hourlyRes] = await Promise.all([
                        fetch(`/analytics/${{venueId}}?days=7`),
                        fetch(`/analytics/${{venueId}}/hourly`)
                    ]);

                    const statsRaw = await statsRes.json();
                    const stats = statsRaw.data || statsRaw;
                    const hourlyRaw = await hourlyRes.json();
                    const hourly = hourlyRaw.data || hourlyRaw;

                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('dashboard').style.display = 'block';

                    // Update stats with clear labels
                    document.getElementById('unique-visitors').textContent = stats.unique_visitors.toLocaleString();
                    document.getElementById('avg-dwell').textContent = stats.avg_dwell_minutes + ' min';
                    document.getElementById('peak-hour').textContent = stats.peak_hour !== null ? stats.peak_hour + ':00' : '-';

                    // Confidence metrics
                    if (stats.confidence_level !== null) {{
                        document.getElementById('confidence').textContent = Math.round(stats.confidence_level * 100) + '%';
                        document.getElementById('data-quality').textContent = 'Data quality: ' + (stats.data_quality || 'unknown');
                        if (stats.visitor_range) {{
                            document.getElementById('visitor-range').textContent =
                                `Range: ${{stats.visitor_range.low}} - ${{stats.visitor_range.high}} (95% CI)`;
                        }}
                    }} else {{
                        document.getElementById('confidence').textContent = '-';
                        document.getElementById('data-quality').textContent = 'No track data';
                    }}

                    // Demographics - only show if we have real data
                    const hasGenderData = Object.keys(stats.gender_split).some(k => k !== null && k !== 'null');
                    const hasAgeData = Object.keys(stats.age_distribution).some(k => k !== null && k !== 'null');

                    if (hasGenderData || hasAgeData) {{
                        document.getElementById('demographics-unavailable').style.display = 'none';
                        document.getElementById('demographics-charts').style.display = 'flex';

                        // Gender chart
                        const genderColors = ['#0066ff', '#ff6b9d'];
                        createPieChart('gender-chart', stats.gender_split, genderColors);
                        createLegend('gender-legend', stats.gender_split, genderColors);

                        // Age chart
                        const ageColors = ['#00cc88', '#0066ff', '#ff9500', '#ff6b6b'];
                        createPieChart('age-chart', stats.age_distribution, ageColors);
                        createLegend('age-legend', stats.age_distribution, ageColors);
                    }} else {{
                        document.getElementById('demographics-unavailable').style.display = 'block';
                        document.getElementById('demographics-charts').style.display = 'none';
                    }}

                    // Hourly chart
                    createBarChart('hourly-chart', hourly.hourly);

                }} catch (e) {{
                    document.getElementById('loading').textContent = 'Error loading data: ' + e.message;
                }}
            }}

            loadData();
        </script>
        </div>
    </body>
    </html>
    """


@router.get("/analytics-dashboard/{venue_id}", response_class=HTMLResponse)
async def analytics_dashboard(venue_id: str):
    """
    Comprehensive analytics dashboard with filters, charts, and data tables.
    """
    venue_id = html.escape(venue_id)
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Analytics Dashboard - {venue_id} - CCTV Analytics</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            * {{ box-sizing: border-box; margin: 0; padding: 0; }}
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #0a0a0a;
                color: #e0e0e0;
                min-height: 100vh;
            }}
            /* Navigation */
            nav {{
                background: #111;
                border-bottom: 1px solid #222;
                padding: 0 20px;
                position: sticky;
                top: 0;
                z-index: 100;
            }}
            nav .nav-inner {{
                max-width: 1600px;
                margin: 0 auto;
                display: flex;
                align-items: center;
                gap: 40px;
                height: 60px;
            }}
            nav .logo {{
                font-size: 20px;
                font-weight: bold;
                color: #fff;
                text-decoration: none;
            }}
            nav .logo span {{ color: #3b82f6; }}
            nav .nav-links {{ display: flex; gap: 30px; }}
            nav .nav-links a {{
                color: #888;
                text-decoration: none;
                font-size: 14px;
                transition: color 0.2s;
            }}
            nav .nav-links a:hover {{ color: #fff; }}
            nav .nav-links a.active {{ color: #3b82f6; }}

            .container {{
                max-width: 1600px;
                margin: 0 auto;
                padding: 30px 20px;
            }}

            /* Header */
            .page-header {{
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                margin-bottom: 30px;
                flex-wrap: wrap;
                gap: 20px;
            }}
            .page-header h1 {{ color: #fff; font-size: 28px; }}
            .page-header .subtitle {{ color: #888; margin-top: 5px; }}

            /* Filters */
            .filters {{
                background: #1a1a1a;
                border: 1px solid #333;
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 25px;
                display: flex;
                flex-wrap: wrap;
                gap: 15px;
                align-items: flex-end;
            }}
            .filter-group {{
                display: flex;
                flex-direction: column;
                gap: 6px;
            }}
            .filter-group label {{
                font-size: 12px;
                color: #888;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            .filter-group select, .filter-group input {{
                background: #0a0a0a;
                border: 1px solid #333;
                border-radius: 6px;
                padding: 10px 14px;
                color: #fff;
                font-size: 14px;
                min-width: 150px;
            }}
            .filter-group select:focus, .filter-group input:focus {{
                border-color: #3b82f6;
                outline: none;
            }}
            .btn {{
                background: #3b82f6;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                font-size: 14px;
                cursor: pointer;
                text-decoration: none;
                display: inline-flex;
                align-items: center;
                gap: 8px;
            }}
            .btn:hover {{ background: #2563eb; }}
            .btn-secondary {{ background: #333; }}
            .btn-secondary:hover {{ background: #444; }}
            .btn-sm {{ padding: 6px 12px; font-size: 12px; }}

            /* KPI Cards */
            .kpi-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 25px;
            }}
            .kpi-card {{
                background: #1a1a1a;
                border: 1px solid #333;
                border-radius: 12px;
                padding: 20px;
            }}
            .kpi-card.highlight {{
                border-color: #3b82f6;
                background: linear-gradient(135deg, #1a1a2e 0%, #1a1a1a 100%);
            }}
            .kpi-value {{
                font-size: 36px;
                font-weight: bold;
                color: #fff;
            }}
            .kpi-card.highlight .kpi-value {{ color: #60a5fa; }}
            .kpi-label {{
                color: #888;
                font-size: 13px;
                margin-top: 5px;
            }}
            .kpi-change {{
                font-size: 12px;
                margin-top: 8px;
                display: flex;
                align-items: center;
                gap: 4px;
            }}
            .kpi-change.positive {{ color: #22c55e; }}
            .kpi-change.negative {{ color: #ef4444; }}
            .kpi-change.neutral {{ color: #888; }}

            /* Charts Grid */
            .charts-grid {{
                display: grid;
                grid-template-columns: 2fr 1fr;
                gap: 20px;
                margin-bottom: 25px;
            }}
            @media (max-width: 1200px) {{
                .charts-grid {{ grid-template-columns: 1fr; }}
            }}

            .card {{
                background: #1a1a1a;
                border: 1px solid #333;
                border-radius: 12px;
                padding: 20px;
            }}
            .card-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
            }}
            .card-header h3 {{
                color: #fff;
                font-size: 16px;
                font-weight: 600;
            }}
            .card-header .tabs {{
                display: flex;
                gap: 5px;
            }}
            .card-header .tab {{
                padding: 6px 12px;
                background: transparent;
                border: 1px solid #333;
                border-radius: 6px;
                color: #888;
                font-size: 12px;
                cursor: pointer;
            }}
            .card-header .tab.active {{
                background: #3b82f6;
                border-color: #3b82f6;
                color: #fff;
            }}

            .chart-container {{
                position: relative;
                height: 300px;
            }}

            /* Side Charts */
            .side-charts {{
                display: flex;
                flex-direction: column;
                gap: 20px;
            }}
            .mini-chart {{
                height: 180px;
            }}

            /* Heatmap */
            .heatmap-container {{
                margin-bottom: 25px;
            }}
            .heatmap {{
                display: grid;
                grid-template-columns: 60px repeat(24, 1fr);
                gap: 2px;
                font-size: 11px;
            }}
            .heatmap-header {{
                color: #666;
                text-align: center;
                padding: 5px 0;
            }}
            .heatmap-row-label {{
                color: #888;
                display: flex;
                align-items: center;
                padding-right: 10px;
            }}
            .heatmap-cell {{
                aspect-ratio: 1;
                border-radius: 3px;
                min-height: 20px;
                cursor: pointer;
                transition: transform 0.1s;
            }}
            .heatmap-cell:hover {{
                transform: scale(1.2);
                z-index: 10;
            }}
            .heatmap-legend {{
                display: flex;
                justify-content: flex-end;
                align-items: center;
                gap: 10px;
                margin-top: 15px;
                font-size: 12px;
                color: #888;
            }}
            .heatmap-legend-gradient {{
                width: 150px;
                height: 12px;
                border-radius: 6px;
                background: linear-gradient(90deg, #1a1a2e, #1e3a5f, #2563eb, #3b82f6, #60a5fa);
            }}

            /* Data Table */
            .table-container {{
                overflow-x: auto;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            th, td {{
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid #333;
            }}
            th {{
                color: #888;
                font-size: 12px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                font-weight: 600;
                cursor: pointer;
                user-select: none;
            }}
            th:hover {{ color: #fff; }}
            th.sorted {{ color: #3b82f6; }}
            td {{ color: #e0e0e0; }}
            tr:hover td {{ background: #222; }}

            .progress-bar {{
                width: 100%;
                height: 6px;
                background: #333;
                border-radius: 3px;
                overflow: hidden;
            }}
            .progress-bar-fill {{
                height: 100%;
                border-radius: 3px;
                transition: width 0.3s;
            }}

            .badge {{
                display: inline-block;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 11px;
                font-weight: 500;
            }}
            .badge-engaged {{ background: #22c55e33; color: #22c55e; }}
            .badge-browsing {{ background: #3b82f633; color: #3b82f6; }}
            .badge-waiting {{ background: #f59e0b33; color: #f59e0b; }}
            .badge-passing {{ background: #6b728033; color: #9ca3af; }}

            /* Loading */
            .loading {{
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 60px;
                color: #888;
            }}
            .spinner {{
                width: 40px;
                height: 40px;
                border: 3px solid #333;
                border-top-color: #3b82f6;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin-right: 15px;
            }}
            @keyframes spin {{ to {{ transform: rotate(360deg); }} }}

            /* Tooltip */
            .tooltip {{
                position: absolute;
                background: #222;
                border: 1px solid #444;
                padding: 8px 12px;
                border-radius: 6px;
                font-size: 12px;
                pointer-events: none;
                z-index: 1000;
                white-space: nowrap;
            }}

            /* Comparison Mode */
            .comparison-toggle {{
                display: flex;
                align-items: center;
                gap: 8px;
                color: #888;
                font-size: 13px;
            }}
            .comparison-toggle input {{
                width: 18px;
                height: 18px;
            }}

            /* Export dropdown */
            .export-dropdown {{
                position: relative;
            }}
            .export-menu {{
                position: absolute;
                top: 100%;
                right: 0;
                background: #222;
                border: 1px solid #333;
                border-radius: 8px;
                padding: 8px 0;
                margin-top: 5px;
                display: none;
                min-width: 150px;
                z-index: 100;
            }}
            .export-menu.show {{ display: block; }}
            .export-menu a {{
                display: block;
                padding: 10px 15px;
                color: #e0e0e0;
                text-decoration: none;
                font-size: 13px;
            }}
            .export-menu a:hover {{ background: #333; }}
        </style>
    </head>
    <body>
        <nav>
            <div class="nav-inner">
                <a href="/" class="logo">CCTV<span>Analytics</span></a>
                <div class="nav-links">
                    <a href="/">Home</a>
                    <a href="/analytics">Analytics</a>
                    <a href="/process">Process Video</a>
                    <a href="/uploads">Batch Upload</a>
                    <a href="/map">Map</a>
                    <a href="/architecture">Architecture</a>
                    <a href="/docs">API Docs</a>
                </div>
            </div>
        </nav>

        <div class="container">
            <!-- Header -->
            <div class="page-header">
                <div>
                    <h1>Analytics Dashboard</h1>
                    <p class="subtitle">Venue: <strong>{venue_id}</strong></p>
                </div>
                <div style="display: flex; gap: 10px; align-items: center;">
                    <label class="comparison-toggle">
                        <input type="checkbox" id="compare-toggle">
                        Compare to previous period
                    </label>
                    <div class="export-dropdown">
                        <button class="btn btn-secondary" onclick="toggleExportMenu()">
                            Export
                            <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor">
                                <path d="M2 4l4 4 4-4"/>
                            </svg>
                        </button>
                        <div class="export-menu" id="export-menu">
                            <a href="/analytics/{venue_id}/export?format=json" target="_blank">Export JSON</a>
                            <a href="/analytics/{venue_id}/export?format=csv">Download CSV</a>
                            <a href="/report/{venue_id}" target="_blank">Print Report</a>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Filters -->
            <div class="filters">
                <div class="filter-group">
                    <label>Date Range</label>
                    <select id="filter-days">
                        <option value="1">Today</option>
                        <option value="7" selected>Last 7 days</option>
                        <option value="14">Last 14 days</option>
                        <option value="30">Last 30 days</option>
                        <option value="90">Last 90 days</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label>Zone</label>
                    <select id="filter-zone">
                        <option value="">All Zones</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label>Gender</label>
                    <select id="filter-gender">
                        <option value="">All</option>
                        <option value="M">Male</option>
                        <option value="F">Female</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label>Age Group</label>
                    <select id="filter-age">
                        <option value="">All Ages</option>
                        <option value="18-24">18-24</option>
                        <option value="25-34">25-34</option>
                        <option value="35-44">35-44</option>
                        <option value="45-54">45-54</option>
                        <option value="55+">55+</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label>Behavior</label>
                    <select id="filter-behavior">
                        <option value="">All Behaviors</option>
                        <option value="engaged">Engaged</option>
                        <option value="browsing">Browsing</option>
                        <option value="waiting">Waiting</option>
                        <option value="passing">Passing</option>
                    </select>
                </div>
                <button class="btn" onclick="applyFilters()">Apply Filters</button>
                <button class="btn btn-secondary" onclick="resetFilters()">Reset</button>
            </div>

            <!-- KPI Cards -->
            <div class="kpi-grid" id="kpi-grid">
                <div class="loading"><div class="spinner"></div> Loading...</div>
            </div>

            <!-- Main Charts -->
            <div class="charts-grid">
                <div class="card">
                    <div class="card-header">
                        <h3>Traffic Trend</h3>
                        <div class="tabs">
                            <button class="tab active" data-view="daily">Daily</button>
                            <button class="tab" data-view="hourly">Hourly</button>
                        </div>
                    </div>
                    <div class="chart-container">
                        <canvas id="traffic-chart"></canvas>
                    </div>
                </div>
                <div class="side-charts">
                    <div class="card">
                        <div class="card-header">
                            <h3>Demographics</h3>
                        </div>
                        <div class="mini-chart">
                            <canvas id="demographics-chart"></canvas>
                        </div>
                    </div>
                    <div class="card">
                        <div class="card-header">
                            <h3>Behavior Mix</h3>
                        </div>
                        <div class="mini-chart">
                            <canvas id="behavior-chart"></canvas>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Heatmap -->
            <div class="card heatmap-container">
                <div class="card-header">
                    <h3>Weekly Traffic Heatmap</h3>
                    <span style="color: #888; font-size: 13px;">Visitors by day and hour</span>
                </div>
                <div class="heatmap" id="heatmap"></div>
                <div class="heatmap-legend">
                    <span>Low</span>
                    <div class="heatmap-legend-gradient"></div>
                    <span>High</span>
                </div>
            </div>

            <!-- Zone Performance Table -->
            <div class="card">
                <div class="card-header">
                    <h3>Zone Performance</h3>
                    <button class="btn btn-sm btn-secondary" onclick="exportTable()">Export CSV</button>
                </div>
                <div class="table-container">
                    <table id="zone-table">
                        <thead>
                            <tr>
                                <th onclick="sortTable(0)">Zone</th>
                                <th onclick="sortTable(1)">Visitors</th>
                                <th onclick="sortTable(2)">Avg Dwell</th>
                                <th onclick="sortTable(3)">Engagement</th>
                                <th>Traffic Share</th>
                                <th onclick="sortTable(5)">Top Behavior</th>
                            </tr>
                        </thead>
                        <tbody id="zone-tbody">
                            <tr><td colspan="6" class="loading"><div class="spinner"></div> Loading...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- Engagement by Hour -->
            <div class="card" style="margin-top: 20px;">
                <div class="card-header">
                    <h3>Engagement by Hour</h3>
                </div>
                <div class="chart-container">
                    <canvas id="engagement-chart"></canvas>
                </div>
            </div>
        </div>

        <script>
            const venueId = '{venue_id}';
            let currentDays = 7;
            let allData = {{}};
            let trafficChart, demographicsChart, behaviorChart, engagementChart;

            // Chart.js defaults
            Chart.defaults.color = '#888';
            Chart.defaults.borderColor = '#333';

            // Initialize
            document.addEventListener('DOMContentLoaded', () => {{
                loadAllData();
                setupEventListeners();
            }});

            function setupEventListeners() {{
                // Tab switching
                document.querySelectorAll('.tab').forEach(tab => {{
                    tab.addEventListener('click', (e) => {{
                        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                        e.target.classList.add('active');
                        updateTrafficChart(e.target.dataset.view);
                    }});
                }});

                // Comparison toggle
                document.getElementById('compare-toggle').addEventListener('change', (e) => {{
                    loadAllData();
                }});
            }}

            async function loadAllData() {{
                const days = document.getElementById('filter-days').value;
                currentDays = parseInt(days);

                try {{
                    const unwrap = r => r.json().then(j => j.data || j);
                    const [summary, hourly, demographics, zones, behavior, behaviorHourly, heatmap] = await Promise.all([
                        fetch(`/analytics/${{venueId}}/summary?days=${{days}}`).then(unwrap),
                        fetch(`/analytics/${{venueId}}/hourly`).then(unwrap),
                        fetch(`/analytics/${{venueId}}/demographics?days=${{days}}`).then(unwrap),
                        fetch(`/analytics/${{venueId}}/zones?days=${{days}}`).then(unwrap),
                        fetch(`/analytics/${{venueId}}/behavior?days=${{days}}`).then(unwrap),
                        fetch(`/analytics/${{venueId}}/behavior/hourly?days=${{days}}`).then(unwrap),
                        fetch(`/analytics/${{venueId}}/heatmap?weeks=${{Math.ceil(days/7)}}`).then(unwrap)
                    ]);

                    allData = {{ summary, hourly, demographics, zones, behavior, behaviorHourly, heatmap }};

                    renderKPIs(summary);
                    renderTrafficChart(hourly);
                    renderDemographicsChart(demographics);
                    renderBehaviorChart(behavior);
                    renderHeatmap(heatmap);
                    renderZoneTable(zones, behavior);
                    renderEngagementChart(behaviorHourly);
                    populateZoneFilter(zones);
                }} catch (error) {{
                    console.error('Error loading data:', error);
                }}
            }}

            function renderKPIs(summary) {{
                const current = summary.current || {{}};
                const change = summary.change || {{}};
                const showComparison = document.getElementById('compare-toggle').checked;

                const kpis = [
                    {{
                        value: current.unique_visitors || 0,
                        label: 'Unique Visitors',
                        change: change.visitors_percent,
                        highlight: true
                    }},
                    {{
                        value: (current.return_rate_percent || 0) + '%',
                        label: 'Return Rate',
                        change: null
                    }},
                    {{
                        value: (current.avg_dwell_minutes || 0) + ' min',
                        label: 'Avg Dwell Time',
                        change: change.dwell_minutes ? (change.dwell_minutes > 0 ? '+' : '') + change.dwell_minutes + ' min' : null
                    }},
                    {{
                        value: current.peak_hour !== null ? current.peak_hour + ':00' : '-',
                        label: 'Peak Hour',
                        change: null
                    }},
                    {{
                        value: current.avg_engagement !== null ? current.avg_engagement : '-',
                        label: 'Avg Engagement',
                        change: null
                    }},
                    {{
                        value: (current.engaged_percent || 0) + '%',
                        label: 'Highly Engaged',
                        change: null
                    }}
                ];

                document.getElementById('kpi-grid').innerHTML = kpis.map(kpi => `
                    <div class="kpi-card ${{kpi.highlight ? 'highlight' : ''}}">
                        <div class="kpi-value">${{kpi.value}}</div>
                        <div class="kpi-label">${{kpi.label}}</div>
                        ${{showComparison && kpi.change !== null ? `
                            <div class="kpi-change ${{kpi.change > 0 ? 'positive' : kpi.change < 0 ? 'negative' : 'neutral'}}">
                                ${{kpi.change > 0 ? '↑' : kpi.change < 0 ? '↓' : '→'}} ${{Math.abs(kpi.change)}}${{typeof kpi.change === 'number' ? '%' : ''}} vs previous
                            </div>
                        ` : ''}}
                    </div>
                `).join('');
            }}

            function renderTrafficChart(hourly) {{
                const ctx = document.getElementById('traffic-chart').getContext('2d');

                if (trafficChart) trafficChart.destroy();

                const data = hourly.hourly || hourly.hourly_breakdown || [];

                trafficChart = new Chart(ctx, {{
                    type: 'line',
                    data: {{
                        labels: data.map(d => d.hour + ':00'),
                        datasets: [{{
                            label: 'Visitors',
                            data: data.map(d => d.visitors),
                            borderColor: '#3b82f6',
                            backgroundColor: 'rgba(59, 130, 246, 0.1)',
                            fill: true,
                            tension: 0.4,
                            pointRadius: 4,
                            pointHoverRadius: 6
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{
                            legend: {{ display: false }}
                        }},
                        scales: {{
                            y: {{
                                beginAtZero: true,
                                grid: {{ color: '#222' }}
                            }},
                            x: {{
                                grid: {{ display: false }}
                            }}
                        }}
                    }}
                }});
            }}

            function renderDemographicsChart(demographics) {{
                const ctx = document.getElementById('demographics-chart').getContext('2d');

                if (demographicsChart) demographicsChart.destroy();

                const genderData = demographics.gender_split || (demographics.current && demographics.current.gender) || {{}};
                const labels = Object.keys(genderData).filter(k => k && k !== 'null' && k !== 'undefined');
                const values = labels.map(k => genderData[k]);

                demographicsChart = new Chart(ctx, {{
                    type: 'doughnut',
                    data: {{
                        labels: labels.map(l => l === 'M' ? 'Male' : l === 'F' ? 'Female' : l),
                        datasets: [{{
                            data: values,
                            backgroundColor: ['#3b82f6', '#ec4899', '#8b5cf6'],
                            borderWidth: 0
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{
                            legend: {{
                                position: 'bottom',
                                labels: {{ padding: 15, usePointStyle: true }}
                            }}
                        }}
                    }}
                }});
            }}

            function renderBehaviorChart(behavior) {{
                const ctx = document.getElementById('behavior-chart').getContext('2d');

                if (behaviorChart) behaviorChart.destroy();

                const types = behavior.behavior_types || {{}};
                const labels = Object.keys(types);
                const values = Object.values(types);

                const colors = {{
                    'engaged': '#22c55e',
                    'browsing': '#3b82f6',
                    'waiting': '#f59e0b',
                    'passing': '#6b7280'
                }};

                behaviorChart = new Chart(ctx, {{
                    type: 'doughnut',
                    data: {{
                        labels: labels.map(l => l.charAt(0).toUpperCase() + l.slice(1)),
                        datasets: [{{
                            data: values,
                            backgroundColor: labels.map(l => colors[l] || '#888'),
                            borderWidth: 0
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{
                            legend: {{
                                position: 'bottom',
                                labels: {{ padding: 15, usePointStyle: true }}
                            }}
                        }}
                    }}
                }});
            }}

            function renderHeatmap(heatmap) {{
                const container = document.getElementById('heatmap');
                const data = heatmap.heatmap || {{}};
                const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
                const maxCount = heatmap.max_count || 1;

                let html = '<div class="heatmap-header"></div>';
                for (let h = 0; h < 24; h++) {{
                    html += `<div class="heatmap-header">${{h}}</div>`;
                }}

                days.forEach(day => {{
                    html += `<div class="heatmap-row-label">${{day}}</div>`;
                    for (let h = 0; h < 24; h++) {{
                        const count = (data[day] && data[day][h]) || 0;
                        const intensity = count / maxCount;
                        const color = getHeatmapColor(intensity);
                        html += `<div class="heatmap-cell" style="background: ${{color}}" title="${{day}} ${{h}}:00 - ${{count}} visitors"></div>`;
                    }}
                }});

                container.innerHTML = html;
            }}

            function getHeatmapColor(intensity) {{
                if (intensity === 0) return '#1a1a2e';
                if (intensity < 0.25) return '#1e3a5f';
                if (intensity < 0.5) return '#2563eb';
                if (intensity < 0.75) return '#3b82f6';
                return '#60a5fa';
            }}

            function renderZoneTable(zones, behavior) {{
                const tbody = document.getElementById('zone-tbody');
                const zoneData = zones.zones || [];
                const behaviorZones = allData.behaviorHourly || {{}};

                if (zoneData.length === 0) {{
                    tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: #666; padding: 40px;">No zone data available</td></tr>';
                    return;
                }}

                const totalVisitors = zoneData.reduce((sum, z) => sum + z.visitors, 0);

                tbody.innerHTML = zoneData.map(zone => {{
                    const share = totalVisitors > 0 ? (zone.visitors / totalVisitors * 100).toFixed(1) : 0;
                    const engagement = zone.engagement || zone.avg_engagement || '-';
                    const topBehavior = zone.top_behavior || 'browsing';

                    return `
                        <tr>
                            <td><strong>${{zone.zone}}</strong></td>
                            <td>${{zone.visitors.toLocaleString()}}</td>
                            <td>${{zone.avg_dwell_minutes || (zone.avg_dwell_seconds / 60).toFixed(1)}} min</td>
                            <td>${{typeof engagement === 'number' ? engagement.toFixed(1) : engagement}}</td>
                            <td>
                                <div style="display: flex; align-items: center; gap: 10px;">
                                    <div class="progress-bar" style="width: 100px;">
                                        <div class="progress-bar-fill" style="width: ${{share}}%; background: #3b82f6;"></div>
                                    </div>
                                    <span>${{share}}%</span>
                                </div>
                            </td>
                            <td><span class="badge badge-${{topBehavior}}">${{topBehavior}}</span></td>
                        </tr>
                    `;
                }}).join('');
            }}

            function renderEngagementChart(behaviorHourly) {{
                const ctx = document.getElementById('engagement-chart').getContext('2d');

                if (engagementChart) engagementChart.destroy();

                const data = behaviorHourly.hourly_engagement || [];

                engagementChart = new Chart(ctx, {{
                    type: 'bar',
                    data: {{
                        labels: data.map(d => d.hour + ':00'),
                        datasets: [{{
                            label: 'Avg Engagement',
                            data: data.map(d => d.avg_engagement),
                            backgroundColor: data.map(d => {{
                                if (d.avg_engagement >= 70) return '#22c55e';
                                if (d.avg_engagement >= 50) return '#3b82f6';
                                return '#6b7280';
                            }}),
                            borderRadius: 4
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{
                            legend: {{ display: false }}
                        }},
                        scales: {{
                            y: {{
                                beginAtZero: true,
                                max: 100,
                                grid: {{ color: '#222' }},
                                title: {{ display: true, text: 'Engagement Score' }}
                            }},
                            x: {{
                                grid: {{ display: false }}
                            }}
                        }}
                    }}
                }});
            }}

            function populateZoneFilter(zones) {{
                const select = document.getElementById('filter-zone');
                const zoneData = zones.zones || [];

                select.innerHTML = '<option value="">All Zones</option>' +
                    zoneData.map(z => `<option value="${{z.zone}}">${{z.zone}}</option>`).join('');
            }}

            function applyFilters() {{
                loadAllData();
            }}

            function resetFilters() {{
                document.getElementById('filter-days').value = '7';
                document.getElementById('filter-zone').value = '';
                document.getElementById('filter-gender').value = '';
                document.getElementById('filter-age').value = '';
                document.getElementById('filter-behavior').value = '';
                loadAllData();
            }}

            function toggleExportMenu() {{
                document.getElementById('export-menu').classList.toggle('show');
            }}

            // Close export menu when clicking outside
            document.addEventListener('click', (e) => {{
                if (!e.target.closest('.export-dropdown')) {{
                    document.getElementById('export-menu').classList.remove('show');
                }}
            }});

            function updateTrafficChart(view) {{
                // For now, just reload - could implement daily aggregation
                renderTrafficChart(allData.hourly);
            }}

            let sortDirection = 1;
            let sortColumn = -1;

            function sortTable(column) {{
                const tbody = document.getElementById('zone-tbody');
                const rows = Array.from(tbody.querySelectorAll('tr'));

                if (sortColumn === column) {{
                    sortDirection *= -1;
                }} else {{
                    sortDirection = 1;
                    sortColumn = column;
                }}

                rows.sort((a, b) => {{
                    let aVal = a.cells[column].textContent.trim();
                    let bVal = b.cells[column].textContent.trim();

                    // Try numeric comparison
                    const aNum = parseFloat(aVal.replace(/[^0-9.-]/g, ''));
                    const bNum = parseFloat(bVal.replace(/[^0-9.-]/g, ''));

                    if (!isNaN(aNum) && !isNaN(bNum)) {{
                        return (aNum - bNum) * sortDirection;
                    }}

                    return aVal.localeCompare(bVal) * sortDirection;
                }});

                rows.forEach(row => tbody.appendChild(row));

                // Update header styling
                document.querySelectorAll('th').forEach((th, i) => {{
                    th.classList.toggle('sorted', i === column);
                }});
            }}

            function exportTable() {{
                const table = document.getElementById('zone-table');
                const rows = table.querySelectorAll('tr');
                let csv = [];

                rows.forEach(row => {{
                    const cols = row.querySelectorAll('th, td');
                    const rowData = Array.from(cols).map(col => {{
                        let text = col.textContent.trim().replace(/"/g, '""');
                        return `"${{text}}"`;
                    }});
                    csv.push(rowData.join(','));
                }});

                const blob = new Blob([csv.join('\\n')], {{ type: 'text/csv' }});
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `${{venueId}}_zone_performance.csv`;
                a.click();
            }}
        </script>
    </body>
    </html>
    """


@router.get("/uploads", response_class=HTMLResponse)
async def uploads_dashboard():
    """Dashboard to manage batch video uploads and queue."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Upload Manager - CCTV Analytics</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #0a0a0a;
                color: #e0e0e0;
                min-height: 100vh;
            }
            /* Navigation */
            nav {
                background: #111;
                border-bottom: 1px solid #222;
                padding: 0 20px;
                position: sticky;
                top: 0;
                z-index: 100;
            }
            nav .nav-inner {
                max-width: 1400px;
                margin: 0 auto;
                display: flex;
                align-items: center;
                gap: 40px;
                height: 60px;
            }
            nav .logo {
                font-size: 20px;
                font-weight: bold;
                color: #fff;
                text-decoration: none;
            }
            nav .logo span { color: #3b82f6; }
            nav .nav-links { display: flex; gap: 30px; }
            nav .nav-links a {
                color: #888;
                text-decoration: none;
                font-size: 14px;
                transition: color 0.2s;
            }
            nav .nav-links a:hover { color: #fff; }
            nav .nav-links a.active { color: #3b82f6; }
            .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
            h1 { color: #fff; margin-bottom: 10px; }
            .subtitle { color: #888; margin-bottom: 30px; }

            /* Stats bar */
            .stats-bar {
                display: flex;
                gap: 20px;
                margin-bottom: 30px;
                flex-wrap: wrap;
            }
            .stat-card {
                background: #1a1a1a;
                border-radius: 12px;
                padding: 20px 30px;
                border: 1px solid #333;
                min-width: 140px;
            }
            .stat-value {
                font-size: 32px;
                font-weight: bold;
                color: #fff;
            }
            .stat-label { color: #888; font-size: 14px; margin-top: 5px; }
            .stat-card.pending .stat-value { color: #f59e0b; }
            .stat-card.processing .stat-value { color: #3b82f6; }
            .stat-card.completed .stat-value { color: #22c55e; }
            .stat-card.failed .stat-value { color: #ef4444; }

            /* Upload section */
            .upload-section {
                background: #1a1a1a;
                border-radius: 12px;
                padding: 30px;
                border: 1px solid #333;
                margin-bottom: 30px;
            }
            .upload-section h2 { margin-bottom: 20px; color: #fff; }

            .upload-area {
                border: 2px dashed #444;
                border-radius: 12px;
                padding: 40px;
                text-align: center;
                cursor: pointer;
                transition: all 0.2s;
                margin-bottom: 20px;
            }
            .upload-area:hover, .upload-area.dragover {
                border-color: #3b82f6;
                background: rgba(59, 130, 246, 0.1);
            }
            .upload-area input { display: none; }
            .upload-icon { font-size: 48px; margin-bottom: 10px; }
            .upload-text { color: #888; }
            .upload-text strong { color: #3b82f6; }

            .upload-options {
                display: flex;
                gap: 15px;
                flex-wrap: wrap;
                align-items: end;
            }
            .form-group { flex: 1; min-width: 200px; }
            .form-group label { display: block; margin-bottom: 8px; color: #888; font-size: 14px; }
            .form-group input, .form-group select {
                width: 100%;
                padding: 12px;
                border-radius: 8px;
                border: 1px solid #333;
                background: #0a0a0a;
                color: #fff;
                font-size: 14px;
            }
            .btn {
                padding: 12px 24px;
                border-radius: 8px;
                border: none;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.2s;
            }
            .btn-primary { background: #3b82f6; color: white; }
            .btn-primary:hover { background: #2563eb; }
            .btn-primary:disabled { background: #555; cursor: not-allowed; }

            /* Jobs table */
            .jobs-section {
                background: #1a1a1a;
                border-radius: 12px;
                padding: 30px;
                border: 1px solid #333;
            }
            .jobs-section h2 { margin-bottom: 20px; color: #fff; }

            .jobs-table {
                width: 100%;
                border-collapse: collapse;
            }
            .jobs-table th, .jobs-table td {
                padding: 15px;
                text-align: left;
                border-bottom: 1px solid #333;
            }
            .jobs-table th { color: #888; font-weight: 500; }
            .jobs-table tr:hover { background: rgba(255,255,255,0.02); }

            .status-badge {
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: 600;
            }
            .status-pending { background: rgba(245, 158, 11, 0.2); color: #f59e0b; }
            .status-processing { background: rgba(59, 130, 246, 0.2); color: #3b82f6; }
            .status-completed { background: rgba(34, 197, 94, 0.2); color: #22c55e; }
            .status-failed { background: rgba(239, 68, 68, 0.2); color: #ef4444; }

            .progress-bar {
                width: 100px;
                height: 6px;
                background: #333;
                border-radius: 3px;
                overflow: hidden;
            }
            .progress-fill {
                height: 100%;
                background: #3b82f6;
                transition: width 0.3s;
            }

            .btn-small {
                padding: 6px 12px;
                font-size: 12px;
            }
            .btn-danger { background: #ef4444; color: white; }
            .btn-danger:hover { background: #dc2626; }

            .empty-state {
                text-align: center;
                padding: 60px 20px;
                color: #666;
            }
            .empty-state .icon { font-size: 48px; margin-bottom: 15px; }

            /* Upload progress list */
            .upload-progress-list {
                margin-top: 20px;
            }
            .upload-item {
                display: flex;
                align-items: center;
                gap: 15px;
                padding: 10px;
                background: #0a0a0a;
                border-radius: 8px;
                margin-bottom: 10px;
            }
            .upload-item .filename { flex: 1; }
            .upload-item .size { color: #888; font-size: 12px; }
        </style>
    </head>
    <body>
        <nav>
            <div class="nav-inner">
                <a href="/" class="logo">CCTV<span>Analytics</span></a>
                <div class="nav-links">
                    <a href="/">Home</a>
                    <a href="/process">Process Video</a>
                    <a href="/uploads" class="active">Batch Upload</a>
                    <a href="/map">Map</a>
                    <a href="/architecture">Architecture</a>
                    <a href="/docs">API Docs</a>
                </div>
            </div>
        </nav>

        <div class="container">
            <h1>📤 Upload Manager</h1>
            <p class="subtitle">Batch upload and process videos</p>

            <!-- Stats Bar -->
            <div class="stats-bar">
                <div class="stat-card pending">
                    <div class="stat-value" id="stat-pending">-</div>
                    <div class="stat-label">Pending</div>
                </div>
                <div class="stat-card processing">
                    <div class="stat-value" id="stat-processing">-</div>
                    <div class="stat-label">Processing</div>
                </div>
                <div class="stat-card completed">
                    <div class="stat-value" id="stat-completed">-</div>
                    <div class="stat-label">Completed</div>
                </div>
                <div class="stat-card failed">
                    <div class="stat-value" id="stat-failed">-</div>
                    <div class="stat-label">Failed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="stat-visitors">-</div>
                    <div class="stat-label">Total Visitors Detected</div>
                </div>
            </div>

            <!-- Upload Section -->
            <div class="upload-section">
                <h2>Upload Videos</h2>
                <div class="upload-area" id="upload-area">
                    <input type="file" id="file-input" multiple accept="video/*">
                    <div class="upload-icon">📁</div>
                    <div class="upload-text">
                        <strong>Click to upload</strong> or drag and drop<br>
                        Multiple video files supported (MP4, MOV, AVI)
                    </div>
                </div>

                <div class="upload-options">
                    <div class="form-group">
                        <label>Venue ID</label>
                        <input type="text" id="venue-id" value="demo_venue" placeholder="Enter venue ID">
                    </div>
                    <div class="form-group">
                        <label>Priority</label>
                        <select id="priority">
                            <option value="0">Normal</option>
                            <option value="5">High</option>
                            <option value="10">Urgent</option>
                        </select>
                    </div>
                    <button class="btn btn-primary" id="upload-btn" disabled>Upload Selected Files</button>
                </div>

                <div class="upload-progress-list" id="upload-progress"></div>
            </div>

            <!-- Jobs Table -->
            <div class="jobs-section">
                <h2>Processing Queue</h2>
                <table class="jobs-table">
                    <thead>
                        <tr>
                            <th>Video</th>
                            <th>Venue</th>
                            <th>Status</th>
                            <th>Progress</th>
                            <th>Visitors</th>
                            <th>Created</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody id="jobs-tbody">
                        <tr>
                            <td colspan="7">
                                <div class="empty-state">
                                    <div class="icon">📭</div>
                                    <div>No jobs yet. Upload some videos to get started!</div>
                                </div>
                            </td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>

        <script>
            let selectedFiles = [];

            // File input handling
            const uploadArea = document.getElementById('upload-area');
            const fileInput = document.getElementById('file-input');
            const uploadBtn = document.getElementById('upload-btn');
            const uploadProgress = document.getElementById('upload-progress');

            uploadArea.addEventListener('click', () => fileInput.click());

            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });

            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('dragover');
            });

            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                handleFiles(e.dataTransfer.files);
            });

            fileInput.addEventListener('change', (e) => {
                handleFiles(e.target.files);
            });

            function handleFiles(files) {
                selectedFiles = Array.from(files).filter(f => f.type.startsWith('video/'));
                updateFileList();
            }

            function updateFileList() {
                if (selectedFiles.length === 0) {
                    uploadProgress.innerHTML = '';
                    uploadBtn.disabled = true;
                    return;
                }

                uploadBtn.disabled = false;
                uploadProgress.innerHTML = selectedFiles.map((f, i) => `
                    <div class="upload-item">
                        <span>📹</span>
                        <span class="filename">${f.name}</span>
                        <span class="size">${(f.size / 1024 / 1024).toFixed(1)} MB</span>
                    </div>
                `).join('');
            }

            uploadBtn.addEventListener('click', async () => {
                if (selectedFiles.length === 0) return;

                uploadBtn.disabled = true;
                uploadBtn.textContent = 'Uploading...';

                const formData = new FormData();
                selectedFiles.forEach(f => formData.append('files', f));
                formData.append('venue_id', document.getElementById('venue-id').value);
                formData.append('priority', document.getElementById('priority').value);

                try {
                    const resp = await fetch('/api/batch/upload', {
                        method: 'POST',
                        body: formData
                    });
                    const raw = await resp.json();
                    const data = raw.data || raw;

                    if (resp.ok) {
                        alert(`Queued ${data.jobs.length} videos for processing!`);
                        selectedFiles = [];
                        updateFileList();
                        fileInput.value = '';
                        loadJobs();
                        loadStats();
                    } else {
                        alert('Upload failed: ' + (data.detail || 'Unknown error'));
                    }
                } catch (e) {
                    alert('Upload failed: ' + e.message);
                } finally {
                    uploadBtn.disabled = false;
                    uploadBtn.textContent = 'Upload Selected Files';
                }
            });

            // Load stats
            async function loadStats() {
                try {
                    const resp = await fetch('/api/batch/stats');
                    const raw = await resp.json();
                    const data = raw.data || raw;

                    document.getElementById('stat-pending').textContent = data.queue.pending;
                    document.getElementById('stat-processing').textContent = data.queue.processing;
                    document.getElementById('stat-completed').textContent = data.queue.completed;
                    document.getElementById('stat-failed').textContent = data.queue.failed;
                    document.getElementById('stat-visitors').textContent = data.total_visitors_detected.toLocaleString();
                } catch (e) {
                    console.error('Failed to load stats:', e);
                }
            }

            // Load jobs
            async function loadJobs() {
                try {
                    const resp = await fetch('/api/batch/jobs?limit=50');
                    const raw = await resp.json();
                    const data = raw.data || raw;

                    const tbody = document.getElementById('jobs-tbody');

                    if (data.jobs.length === 0) {
                        tbody.innerHTML = `
                            <tr>
                                <td colspan="7">
                                    <div class="empty-state">
                                        <div class="icon">📭</div>
                                        <div>No jobs yet. Upload some videos to get started!</div>
                                    </div>
                                </td>
                            </tr>
                        `;
                        return;
                    }

                    tbody.innerHTML = data.jobs.map(job => `
                        <tr data-job-id="${job.id}">
                            <td title="${job.video_name}">${truncate(job.video_name, 30)}</td>
                            <td><a href="/analytics-dashboard/${job.venue_id}">${job.venue_id}</a></td>
                            <td><span class="status-badge status-${job.status}">${job.status}</span></td>
                            <td>
                                ${job.status === 'processing' ? `
                                    <div class="progress-bar">
                                        <div class="progress-fill" style="width: ${job.progress}%"></div>
                                    </div>
                                ` : job.status === 'completed' ? '100%' : '-'}
                            </td>
                            <td>${job.visitors_detected || '-'}</td>
                            <td>${formatTime(job.created_at)}</td>
                            <td>
                                ${job.status === 'pending' ? `
                                    <button class="btn btn-small btn-danger" onclick="cancelJob('${job.id}')">Cancel</button>
                                ` : job.status === 'failed' ? `
                                    <span title="${job.error_message || 'Unknown error'}">⚠️</span>
                                ` : ''}
                            </td>
                        </tr>
                    `).join('');
                } catch (e) {
                    console.error('Failed to load jobs:', e);
                }
            }

            function truncate(str, len) {
                if (!str) return '-';
                return str.length > len ? str.slice(0, len) + '...' : str;
            }

            function formatTime(iso) {
                if (!iso) return '-';
                const d = new Date(iso);
                return d.toLocaleString();
            }

            async function cancelJob(jobId) {
                if (!confirm('Cancel this job?')) return;

                try {
                    const resp = await fetch(`/api/batch/jobs/${jobId}`, { method: 'DELETE' });
                    if (resp.ok) {
                        loadJobs();
                        loadStats();
                    } else {
                        const data = await resp.json();
                        alert('Failed to cancel: ' + (data.detail || 'Unknown error'));
                    }
                } catch (e) {
                    alert('Failed to cancel: ' + e.message);
                }
            }

            // Initial load and auto-refresh
            loadStats();
            loadJobs();
            setInterval(() => {
                loadStats();
                loadJobs();
            }, 3000);  // Refresh every 3 seconds
        </script>
    </body>
    </html>
    """


@router.get("/map", response_class=HTMLResponse)
async def map_view():
    """Interactive map showing all venues with analytics."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Venue Map - CCTV Analytics</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #0a0a0a;
                color: #e0e0e0;
            }
            /* Navigation */
            nav {
                background: #111;
                border-bottom: 1px solid #222;
                padding: 0 20px;
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                z-index: 1001;
            }
            nav .nav-inner {
                max-width: 1400px;
                margin: 0 auto;
                display: flex;
                align-items: center;
                gap: 40px;
                height: 60px;
            }
            nav .logo {
                font-size: 20px;
                font-weight: bold;
                color: #fff;
                text-decoration: none;
            }
            nav .logo span { color: #3b82f6; }
            nav .nav-links { display: flex; gap: 30px; }
            nav .nav-links a {
                color: #888;
                text-decoration: none;
                font-size: 14px;
                transition: color 0.2s;
            }
            nav .nav-links a:hover { color: #fff; }
            nav .nav-links a.active { color: #3b82f6; }
            #map { width: 100%; height: calc(100vh - 60px); margin-top: 60px; }
            .legend {
                position: absolute;
                bottom: 30px;
                right: 10px;
                z-index: 1000;
                background: rgba(0,0,0,0.8);
                padding: 15px;
                border-radius: 8px;
            }
            .legend h4 { margin-bottom: 10px; color: #fff; font-size: 14px; }
            .legend-item {
                display: flex;
                align-items: center;
                gap: 8px;
                margin: 5px 0;
                font-size: 12px;
            }
            .legend-color {
                width: 16px;
                height: 16px;
                border-radius: 50%;
            }
            .venue-popup h3 { margin: 0 0 10px 0; color: #333; }
            .venue-popup p { margin: 5px 0; color: #666; font-size: 13px; }
            .venue-popup .stat { font-weight: bold; color: #0066ff; }
            .venue-popup a {
                display: inline-block;
                margin-top: 10px;
                padding: 8px 16px;
                background: #0066ff;
                color: white;
                text-decoration: none;
                border-radius: 4px;
                font-size: 12px;
            }
            .stats-panel {
                position: absolute;
                top: 80px;
                left: 10px;
                z-index: 1000;
                background: rgba(0,0,0,0.8);
                padding: 15px;
                border-radius: 8px;
                min-width: 200px;
            }
            .stats-panel h4 { margin-bottom: 10px; color: #fff; }
            .stats-panel .stat-row {
                display: flex;
                justify-content: space-between;
                margin: 8px 0;
                font-size: 13px;
            }
            .stats-panel .stat-value { color: #3b82f6; font-weight: bold; }
        </style>
    </head>
    <body>
        <nav>
            <div class="nav-inner">
                <a href="/" class="logo">CCTV<span>Analytics</span></a>
                <div class="nav-links">
                    <a href="/">Home</a>
                    <a href="/process">Process Video</a>
                    <a href="/uploads">Batch Upload</a>
                    <a href="/map" class="active">Map</a>
                    <a href="/docs">API Docs</a>
                </div>
            </div>
        </nav>

        <div id="map"></div>

        <div class="stats-panel" id="stats-panel">
            <h4>🗺️ Overview</h4>
            <div class="stat-row">
                <span>Total Venues</span>
                <span class="stat-value" id="total-venues">-</span>
            </div>
            <div class="stat-row">
                <span>Total Visitors</span>
                <span class="stat-value" id="total-visitors">-</span>
            </div>
            <div class="stat-row">
                <span>Countries</span>
                <span class="stat-value" id="total-countries">-</span>
            </div>
        </div>

        <div class="legend">
            <h4>Traffic Level</h4>
            <div class="legend-item">
                <div class="legend-color" style="background: #22c55e;"></div>
                <span>High (50+ visitors)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #eab308;"></div>
                <span>Medium (10-50)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #3b82f6;"></div>
                <span>Low (< 10)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #6b7280;"></div>
                <span>No data</span>
            </div>
        </div>

        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <script>
            // Initialize map centered on Africa with better styling
            const map = L.map('map', {
                zoomControl: false  // We'll add custom position
            }).setView([-1.2921, 20.0], 3);

            // Add zoom control to bottom right
            L.control.zoom({ position: 'bottomleft' }).addTo(map);

            // Use Stadia Alidade Smooth Dark (free, beautiful)
            L.tileLayer('https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{z}/{x}/{y}{r}.png', {
                maxZoom: 20,
                attribution: '&copy; Stadia Maps, &copy; OpenMapTiles, &copy; OpenStreetMap'
            }).addTo(map);

            // Fetch venues and analytics
            async function loadVenues() {
                try {
                    const response = await fetch('/api/map/venues');
                    const raw = await response.json();
                    const data = raw.data || raw;

                    // Update stats
                    document.getElementById('total-venues').textContent = data.venues.length;
                    document.getElementById('total-visitors').textContent =
                        data.venues.reduce((sum, v) => sum + (v.visitors || 0), 0).toLocaleString();

                    const countries = new Set(data.venues.map(v => v.country).filter(c => c));
                    document.getElementById('total-countries').textContent = countries.size || '-';

                    // Add markers with pulse animation for high traffic
                    data.venues.forEach(venue => {
                        if (venue.latitude && venue.longitude) {
                            const color = getMarkerColor(venue.visitors);
                            const size = Math.max(10, Math.min(25, (venue.visitors || 0) / 3 + 10));

                            // Create custom icon with glow effect
                            const marker = L.circleMarker([venue.latitude, venue.longitude], {
                                radius: size,
                                fillColor: color,
                                color: color,
                                weight: 3,
                                opacity: 0.3,
                                fillOpacity: 0.9
                            }).addTo(map);

                            // Add inner dot
                            L.circleMarker([venue.latitude, venue.longitude], {
                                radius: size * 0.4,
                                fillColor: '#fff',
                                color: '#fff',
                                weight: 0,
                                fillOpacity: 0.9
                            }).addTo(map);

                            const venueTypeIcon = {
                                'bar': '🍺', 'restaurant': '🍽️', 'cafe': '☕',
                                'retail': '🛍️', 'nightclub': '🎵', 'hotel': '🏨',
                                'mall': '🏬', 'other': '📍'
                            }[venue.venue_type] || '📍';

                            marker.bindPopup(`
                                <div class="venue-popup">
                                    <h3>${venueTypeIcon} ${venue.name || venue.id}</h3>
                                    <p><strong>Type:</strong> ${venue.venue_type || 'Unknown'}</p>
                                    <p><strong>Location:</strong> ${[venue.city, venue.country].filter(x=>x).join(', ') || 'Not set'}</p>
                                    <hr style="border: none; border-top: 1px solid #eee; margin: 10px 0;">
                                    <p><strong>Total Visitors:</strong> <span class="stat">${(venue.visitors || 0).toLocaleString()}</span></p>
                                    <p><strong>Zone ID:</strong> <code style="background:#f0f0f0;padding:2px 6px;border-radius:3px;font-size:11px;">${venue.h3_zone || 'N/A'}</code></p>
                                    <a href="/analytics-dashboard/${venue.id}">View Full Analytics →</a>
                                </div>
                            `, { maxWidth: 300 });
                        }
                    });

                    // Fit bounds if we have venues
                    const venuesWithLocation = data.venues.filter(v => v.latitude && v.longitude);
                    if (venuesWithLocation.length > 0) {
                        const bounds = L.latLngBounds(
                            venuesWithLocation.map(v => [v.latitude, v.longitude])
                        );
                        map.fitBounds(bounds, { padding: [50, 50] });
                    }

                } catch (e) {
                    console.error('Failed to load venues:', e);
                }
            }

            function getMarkerColor(visitors) {
                if (!visitors) return '#6b7280';  // Gray - no data
                if (visitors >= 50) return '#22c55e';  // Green - high
                if (visitors >= 10) return '#eab308';  // Yellow - medium
                return '#3b82f6';  // Blue - low
            }

            loadVenues();
        </script>
    </body>
    </html>
    """


@router.get("/architecture", response_class=HTMLResponse)
async def architecture_page():
    """Interactive system architecture schematic."""
    return _ARCHITECTURE_HTML


_ARCHITECTURE_HTML = r"""
<!DOCTYPE html>
<html>
<head>
<title>CCTV Analytics - Architecture</title>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0a0a0a;color:#e0e0e0;min-height:100vh}
nav{background:#111;border-bottom:1px solid #222;padding:0 20px;position:sticky;top:0;z-index:100}
nav .ni{max-width:1600px;margin:0 auto;display:flex;align-items:center;gap:40px;height:60px}
nav .logo{font-size:20px;font-weight:bold;color:#fff;text-decoration:none}
nav .logo span{color:#3b82f6}
nav .nl{display:flex;gap:30px}
nav a{color:#888;text-decoration:none;font-size:14px;transition:color .2s}
nav a:hover{color:#fff}
nav a.active{color:#3b82f6}
.ph{text-align:center;padding:30px 20px 10px}
.ph h1{font-size:28px;color:#fff}
.ph p{color:#888;margin-top:6px;font-size:14px}
.leg{display:flex;justify-content:center;gap:24px;padding:12px 20px;flex-wrap:wrap}
.li{display:flex;align-items:center;gap:8px;font-size:12px;color:#aaa}
.ld{width:14px;height:14px;border-radius:3px}
.ac{max-width:1600px;margin:0 auto;padding:10px 20px 40px;overflow-x:auto}
svg{display:block;margin:0 auto}
.tt{position:fixed;background:#1a1a2e;border:1px solid #3b82f6;border-radius:8px;padding:12px 16px;font-size:13px;color:#e0e0e0;max-width:320px;pointer-events:none;opacity:0;transition:opacity .15s;z-index:200;line-height:1.5}
.tt .t1{font-weight:600;color:#fff;margin-bottom:4px;font-size:14px}
.tt .t2{color:#3b82f6;font-family:monospace;font-size:11px}
.dp{max-width:1600px;margin:0 auto;padding:0 20px 40px}
.dc{background:#111;border:1px solid #222;border-radius:8px;padding:20px;display:none}
.dc.a{display:block}
.dc h3{color:#3b82f6;margin-bottom:8px}
</style>
</head>
<body>
<nav><div class="ni">
<a href="/" class="logo">SIPP <span>Analytics</span></a>
<div class="nl">
<a href="/">Home</a><a href="/analytics">Analytics</a><a href="/process">Process Video</a>
<a href="/uploads">Batch Upload</a><a href="/map">Map</a>
<a href="/architecture" class="active">Architecture</a><a href="/docs">API Docs</a>
</div></div></nav>

<div class="ph"><h1>System Architecture</h1><p>Hover nodes for details &middot; Click for full description</p></div>
<div class="leg">
<div class="li"><div class="ld" style="background:#3b82f6"></div>Input</div>
<div class="li"><div class="ld" style="background:#8b5cf6"></div>Processing</div>
<div class="li"><div class="ld" style="background:#f59e0b"></div>ML / AI</div>
<div class="li"><div class="ld" style="background:#10b981"></div>Learning</div>
<div class="li"><div class="ld" style="background:#ef4444"></div>Output</div>
<div class="li"><div class="ld" style="background:#64748b"></div>Storage</div>
</div>

<div class="ac">
<svg viewBox="0 0 1520 1020" width="1520" height="1020" xmlns="http://www.w3.org/2000/svg">
<defs>
<marker id="a" viewBox="0 0 10 6" refX="10" refY="3" markerWidth="8" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 3L0 6z" fill="#555"/></marker>
<marker id="ab" viewBox="0 0 10 6" refX="10" refY="3" markerWidth="8" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 3L0 6z" fill="#3b82f6"/></marker>
</defs>

<!-- LANES -->
<rect x="10" y="40" width="290" height="950" rx="12" fill="#0d1117" stroke="#1e3a5f"/>
<rect x="310" y="40" width="290" height="950" rx="12" fill="#0d1117" stroke="#2d1b69"/>
<rect x="610" y="40" width="290" height="950" rx="12" fill="#0d1117" stroke="#5c3d0a"/>
<rect x="910" y="40" width="290" height="950" rx="12" fill="#0d1117" stroke="#064e3b"/>
<rect x="1210" y="40" width="300" height="950" rx="12" fill="#0d1117" stroke="#5c1a1a"/>

<text x="155" y="70" text-anchor="middle" fill="#3b82f6" font-size="15" font-weight="bold">INPUT</text>
<text x="455" y="70" text-anchor="middle" fill="#8b5cf6" font-size="15" font-weight="bold">PROCESSING</text>
<text x="755" y="70" text-anchor="middle" fill="#f59e0b" font-size="15" font-weight="bold">ML / AI</text>
<text x="1055" y="70" text-anchor="middle" fill="#10b981" font-size="15" font-weight="bold">LEARNING</text>
<text x="1360" y="70" text-anchor="middle" fill="#ef4444" font-size="15" font-weight="bold">OUTPUT</text>

<!-- ══ ROW 1: MVP ══ -->
<text x="760" y="103" text-anchor="middle" fill="#555" font-size="11" letter-spacing="3">MVP VIDEO PIPELINE</text>

<rect class="n" data-id="video-src" x="30" y="118" width="250" height="48" rx="8" fill="#111827" stroke="#3b82f6" stroke-width="1.5"/>
<text x="155" y="138" text-anchor="middle" fill="#fff" font-size="13" font-weight="600">Video Source</text>
<text x="155" y="155" text-anchor="middle" fill="#888" font-size="10">YouTube / Upload / RTSP</text>
<rect class="n" data-id="download" x="30" y="178" width="250" height="38" rx="8" fill="#111827" stroke="#3b82f6" stroke-width="1.5"/>
<text x="155" y="202" text-anchor="middle" fill="#fff" font-size="12">yt-dlp Download</text>

<rect class="n" data-id="pipeline" x="330" y="118" width="250" height="48" rx="8" fill="#1a1033" stroke="#8b5cf6" stroke-width="1.5"/>
<text x="455" y="138" text-anchor="middle" fill="#fff" font-size="13" font-weight="600">Video Pipeline</text>
<text x="455" y="155" text-anchor="middle" fill="#888" font-size="10">Frame loop + tracking</text>
<rect class="n" data-id="queue" x="330" y="178" width="250" height="38" rx="8" fill="#1a1033" stroke="#8b5cf6" stroke-width="1.5"/>
<text x="455" y="202" text-anchor="middle" fill="#fff" font-size="12">Job Queue</text>

<rect class="n" data-id="yolo-mvp" x="630" y="118" width="250" height="48" rx="8" fill="#1a1505" stroke="#f59e0b" stroke-width="1.5"/>
<text x="755" y="138" text-anchor="middle" fill="#fff" font-size="13" font-weight="600">YOLO11s</text>
<text x="755" y="155" text-anchor="middle" fill="#888" font-size="10">Person detection + BoT-SORT</text>
<rect class="n" data-id="insightface" x="630" y="178" width="250" height="38" rx="8" fill="#1a1505" stroke="#f59e0b" stroke-width="1.5"/>
<text x="755" y="202" text-anchor="middle" fill="#fff" font-size="12">InsightFace buffalo_l</text>
<rect class="n" data-id="yolo-pose" x="630" y="228" width="250" height="38" rx="8" fill="#1a1505" stroke="#f59e0b" stroke-width="1.5"/>
<text x="755" y="252" text-anchor="middle" fill="#fff" font-size="12">YOLO11s-Pose (behavior)</text>

<rect class="n" data-id="face-embed" x="930" y="118" width="250" height="48" rx="8" fill="#051a14" stroke="#10b981" stroke-width="1.5"/>
<text x="1055" y="138" text-anchor="middle" fill="#fff" font-size="13" font-weight="600">Face Embeddings</text>
<text x="1055" y="155" text-anchor="middle" fill="#888" font-size="10">512-d cosine dedup + ReID</text>
<rect class="n" data-id="behavior" x="930" y="178" width="250" height="38" rx="8" fill="#051a14" stroke="#10b981" stroke-width="1.5"/>
<text x="1055" y="202" text-anchor="middle" fill="#fff" font-size="12">Behavior Analysis</text>
<rect class="n" data-id="demographics" x="930" y="228" width="250" height="38" rx="8" fill="#051a14" stroke="#10b981" stroke-width="1.5"/>
<text x="1055" y="252" text-anchor="middle" fill="#fff" font-size="12">Demographics (age/gender)</text>

<rect class="n" data-id="fastapi" x="1230" y="118" width="260" height="48" rx="8" fill="#1a0a0a" stroke="#ef4444" stroke-width="1.5"/>
<text x="1360" y="138" text-anchor="middle" fill="#fff" font-size="13" font-weight="600">FastAPI Server</text>
<text x="1360" y="155" text-anchor="middle" fill="#888" font-size="10">13 routers, 37 endpoints</text>
<rect class="n" data-id="sqlite" x="1230" y="178" width="260" height="38" rx="8" fill="#1a1a1a" stroke="#64748b" stroke-width="1.5"/>
<text x="1360" y="202" text-anchor="middle" fill="#94a3b8" font-size="12">SQLite + WAL (5 tables)</text>
<rect class="n" data-id="html-ui" x="1230" y="228" width="260" height="38" rx="8" fill="#1a0a0a" stroke="#ef4444" stroke-width="1.5"/>
<text x="1360" y="252" text-anchor="middle" fill="#fff" font-size="12">Web Dashboard (8 pages)</text>

<!-- R1 arrows -->
<line x1="280" y1="142" x2="328" y2="142" stroke="#555" stroke-width="1.5" marker-end="url(#a)"/>
<line x1="280" y1="197" x2="328" y2="197" stroke="#555" stroke-width="1.5" marker-end="url(#a)"/>
<line x1="580" y1="142" x2="628" y2="142" stroke="#555" stroke-width="1.5" marker-end="url(#a)"/>
<line x1="580" y1="197" x2="628" y2="197" stroke="#555" stroke-width="1.5" marker-end="url(#a)"/>
<path d="M580 142L600 142L600 247L628 247" fill="none" stroke="#555" stroke-width="1" marker-end="url(#a)"/>
<line x1="880" y1="142" x2="928" y2="142" stroke="#555" stroke-width="1.5" marker-end="url(#a)"/>
<line x1="880" y1="197" x2="928" y2="197" stroke="#555" stroke-width="1.5" marker-end="url(#a)"/>
<line x1="880" y1="247" x2="928" y2="247" stroke="#555" stroke-width="1.5" marker-end="url(#a)"/>
<line x1="1180" y1="142" x2="1228" y2="142" stroke="#555" stroke-width="1.5" marker-end="url(#a)"/>
<line x1="1180" y1="197" x2="1228" y2="197" stroke="#555" stroke-width="1.5" marker-end="url(#a)"/>
<line x1="1180" y1="247" x2="1228" y2="247" stroke="#555" stroke-width="1.5" marker-end="url(#a)"/>

<!-- ══ ROW 2: SIPP ══ -->
<text x="760" y="305" text-anchor="middle" fill="#555" font-size="11" letter-spacing="3">SIPP BAR DETECTION PIPELINE</text>

<rect class="n" data-id="rtsp" x="30" y="322" width="250" height="48" rx="8" fill="#111827" stroke="#3b82f6" stroke-width="1.5"/>
<text x="155" y="342" text-anchor="middle" fill="#fff" font-size="13" font-weight="600">RTSP / Video Feed</text>
<text x="155" y="359" text-anchor="middle" fill="#888" font-size="10">Live camera or file</text>
<rect class="n" data-id="zones-cfg" x="30" y="382" width="250" height="38" rx="8" fill="#111827" stroke="#3b82f6" stroke-width="1.5"/>
<text x="155" y="406" text-anchor="middle" fill="#fff" font-size="12">zones.json + settings.py</text>

<rect class="n" data-id="dual-tracker" x="330" y="322" width="250" height="48" rx="8" fill="#1a1033" stroke="#8b5cf6" stroke-width="1.5"/>
<text x="455" y="342" text-anchor="middle" fill="#fff" font-size="13" font-weight="600">DualTracker</text>
<text x="455" y="359" text-anchor="middle" fill="#888" font-size="10">BoT-SORT + ByteTrack</text>
<rect class="n" data-id="interactions" x="330" y="382" width="250" height="38" rx="8" fill="#1a1033" stroke="#8b5cf6" stroke-width="1.5"/>
<text x="455" y="406" text-anchor="middle" fill="#fff" font-size="12">InteractionDetector</text>
<rect class="n" data-id="clip-buf" x="330" y="432" width="250" height="38" rx="8" fill="#1a1033" stroke="#8b5cf6" stroke-width="1.5"/>
<text x="455" y="456" text-anchor="middle" fill="#fff" font-size="12">ClipBuffer (ring buffer)</text>

<rect class="n" data-id="yolo-bar" x="630" y="322" width="250" height="48" rx="8" fill="#1a1505" stroke="#f59e0b" stroke-width="1.5"/>
<text x="755" y="342" text-anchor="middle" fill="#fff" font-size="13" font-weight="600">YOLO11s / bar</text>
<text x="755" y="359" text-anchor="middle" fill="#888" font-size="10">17-class or COCO auto-switch</text>
<rect class="n" data-id="vlm" x="630" y="382" width="250" height="48" rx="8" fill="#1a1505" stroke="#f59e0b" stroke-width="1.5"/>
<text x="755" y="402" text-anchor="middle" fill="#fff" font-size="13" font-weight="600">Claude Vision API</text>
<text x="755" y="419" text-anchor="middle" fill="#888" font-size="10">9 action classes</text>
<rect class="n" data-id="safe-parse" x="630" y="442" width="250" height="38" rx="8" fill="#1a1505" stroke="#f59e0b" stroke-width="1.5"/>
<text x="755" y="466" text-anchor="middle" fill="#fff" font-size="12">_safe_parse + budget cap</text>

<rect class="n" data-id="dedup" x="930" y="322" width="250" height="48" rx="8" fill="#051a14" stroke="#10b981" stroke-width="1.5"/>
<text x="1055" y="342" text-anchor="middle" fill="#fff" font-size="13" font-weight="600">EventDeduplicator</text>
<text x="1055" y="359" text-anchor="middle" fill="#888" font-size="10">10s sliding window</text>
<rect class="n" data-id="cooldown" x="930" y="382" width="250" height="38" rx="8" fill="#051a14" stroke="#10b981" stroke-width="1.5"/>
<text x="1055" y="406" text-anchor="middle" fill="#fff" font-size="12">Cooldown + Velocity Gate</text>
<rect class="n" data-id="budget" x="930" y="432" width="250" height="38" rx="8" fill="#051a14" stroke="#10b981" stroke-width="1.5"/>
<text x="1055" y="456" text-anchor="middle" fill="#fff" font-size="12">VLM Budget (100/hr)</text>

<rect class="n" data-id="bar-event" x="1230" y="322" width="260" height="48" rx="8" fill="#1a0a0a" stroke="#ef4444" stroke-width="1.5"/>
<text x="1360" y="342" text-anchor="middle" fill="#fff" font-size="13" font-weight="600">BarEvent</text>
<text x="1360" y="359" text-anchor="middle" fill="#888" font-size="10">serve / payment / activity</text>
<rect class="n" data-id="event-api" x="1230" y="382" width="260" height="38" rx="8" fill="#1a0a0a" stroke="#ef4444" stroke-width="1.5"/>
<text x="1360" y="406" text-anchor="middle" fill="#fff" font-size="12">POST /events/batch</text>
<rect class="n" data-id="clips" x="1230" y="432" width="260" height="38" rx="8" fill="#1a1a1a" stroke="#64748b" stroke-width="1.5"/>
<text x="1360" y="456" text-anchor="middle" fill="#94a3b8" font-size="12">confirmed_clips/ (training)</text>

<!-- R2 arrows -->
<line x1="280" y1="346" x2="328" y2="346" stroke="#555" stroke-width="1.5" marker-end="url(#a)"/>
<line x1="280" y1="401" x2="328" y2="401" stroke="#555" stroke-width="1.5" marker-end="url(#a)"/>
<line x1="580" y1="346" x2="628" y2="346" stroke="#555" stroke-width="1.5" marker-end="url(#a)"/>
<line x1="580" y1="401" x2="628" y2="401" stroke="#555" stroke-width="1.5" marker-end="url(#a)"/>
<line x1="580" y1="451" x2="628" y2="410" stroke="#555" stroke-width="1" marker-end="url(#a)"/>
<line x1="880" y1="346" x2="928" y2="346" stroke="#555" stroke-width="1.5" marker-end="url(#a)"/>
<line x1="880" y1="410" x2="928" y2="401" stroke="#555" stroke-width="1.5" marker-end="url(#a)"/>
<line x1="880" y1="461" x2="928" y2="451" stroke="#555" stroke-width="1" marker-end="url(#a)"/>
<line x1="1180" y1="346" x2="1228" y2="346" stroke="#555" stroke-width="1.5" marker-end="url(#a)"/>
<line x1="1180" y1="401" x2="1228" y2="401" stroke="#555" stroke-width="1.5" marker-end="url(#a)"/>
<line x1="1180" y1="451" x2="1228" y2="451" stroke="#555" stroke-width="1.5" marker-end="url(#a)"/>

<!-- ══ ROW 3: TRAINING ══ -->
<text x="760" y="515" text-anchor="middle" fill="#555" font-size="11" letter-spacing="3">BAR MODEL TRAINING PIPELINE</text>

<rect class="n" data-id="raw-video" x="30" y="532" width="250" height="48" rx="8" fill="#111827" stroke="#3b82f6" stroke-width="1.5"/>
<text x="155" y="552" text-anchor="middle" fill="#fff" font-size="13" font-weight="600">Raw Bar Footage</text>
<text x="155" y="569" text-anchor="middle" fill="#888" font-size="10">+ confirmed_clips/</text>
<rect class="n" data-id="extract" x="30" y="592" width="250" height="38" rx="8" fill="#111827" stroke="#3b82f6" stroke-width="1.5"/>
<text x="155" y="616" text-anchor="middle" fill="#fff" font-size="12">extract_frames.py (1 FPS)</text>

<rect class="n" data-id="autolabel" x="330" y="532" width="250" height="48" rx="8" fill="#1a1033" stroke="#8b5cf6" stroke-width="1.5"/>
<text x="455" y="552" text-anchor="middle" fill="#fff" font-size="13" font-weight="600">Grounding DINO</text>
<text x="455" y="569" text-anchor="middle" fill="#888" font-size="10">Autodistill, 16 prompts</text>
<rect class="n" data-id="sahi" x="330" y="592" width="250" height="38" rx="8" fill="#1a1033" stroke="#8b5cf6" stroke-width="1.5"/>
<text x="455" y="616" text-anchor="middle" fill="#fff" font-size="12">SAHI Sliced Inference</text>
<rect class="n" data-id="remap" x="330" y="642" width="250" height="38" rx="8" fill="#1a1033" stroke="#8b5cf6" stroke-width="1.5"/>
<text x="455" y="666" text-anchor="middle" fill="#fff" font-size="12">COCO Remap (4 classes)</text>

<rect class="n" data-id="merge" x="630" y="532" width="250" height="48" rx="8" fill="#1a1505" stroke="#f59e0b" stroke-width="1.5"/>
<text x="755" y="552" text-anchor="middle" fill="#fff" font-size="13" font-weight="600">Merge + NMS Dedup</text>
<text x="755" y="569" text-anchor="middle" fill="#888" font-size="10">3 sources unified</text>
<rect class="n" data-id="train" x="630" y="592" width="250" height="48" rx="8" fill="#1a1505" stroke="#f59e0b" stroke-width="1.5"/>
<text x="755" y="612" text-anchor="middle" fill="#fff" font-size="13" font-weight="600">YOLO11s Fine-tune</text>
<text x="755" y="629" text-anchor="middle" fill="#888" font-size="10">50 epochs, 640px</text>

<rect class="n" data-id="evaluate" x="930" y="532" width="250" height="48" rx="8" fill="#051a14" stroke="#10b981" stroke-width="1.5"/>
<text x="1055" y="552" text-anchor="middle" fill="#fff" font-size="13" font-weight="600">Evaluate</text>
<text x="1055" y="569" text-anchor="middle" fill="#888" font-size="10">bar vs COCO baseline</text>
<rect class="n" data-id="ontology" x="930" y="592" width="250" height="38" rx="8" fill="#051a14" stroke="#10b981" stroke-width="1.5"/>
<text x="1055" y="616" text-anchor="middle" fill="#fff" font-size="12">17-Class Bar Ontology</text>

<rect class="n" data-id="bar-model" x="1230" y="532" width="260" height="48" rx="8" fill="#1a0a0a" stroke="#ef4444" stroke-width="1.5"/>
<text x="1360" y="552" text-anchor="middle" fill="#fff" font-size="13" font-weight="600">yolo11s-bar.pt</text>
<text x="1360" y="569" text-anchor="middle" fill="#888" font-size="10">Fine-tuned bar model</text>
<rect class="n" data-id="data-yaml" x="1230" y="592" width="260" height="38" rx="8" fill="#1a1a1a" stroke="#64748b" stroke-width="1.5"/>
<text x="1360" y="616" text-anchor="middle" fill="#94a3b8" font-size="12">data.yaml (17 classes)</text>

<!-- R3 arrows -->
<line x1="280" y1="556" x2="328" y2="556" stroke="#555" stroke-width="1.5" marker-end="url(#a)"/>
<line x1="280" y1="611" x2="328" y2="611" stroke="#555" stroke-width="1.5" marker-end="url(#a)"/>
<line x1="580" y1="556" x2="628" y2="556" stroke="#555" stroke-width="1.5" marker-end="url(#a)"/>
<line x1="580" y1="611" x2="628" y2="565" stroke="#555" stroke-width="1" marker-end="url(#a)"/>
<line x1="580" y1="661" x2="628" y2="565" stroke="#555" stroke-width="1" marker-end="url(#a)"/>
<line x1="880" y1="556" x2="928" y2="556" stroke="#555" stroke-width="1.5" marker-end="url(#a)"/>
<line x1="880" y1="616" x2="928" y2="611" stroke="#555" stroke-width="1.5" marker-end="url(#a)"/>
<line x1="1180" y1="556" x2="1228" y2="556" stroke="#555" stroke-width="1.5" marker-end="url(#a)"/>
<line x1="1180" y1="611" x2="1228" y2="611" stroke="#555" stroke-width="1.5" marker-end="url(#a)"/>

<!-- ══ ROW 4: HARDENING ══ -->
<text x="760" y="710" text-anchor="middle" fill="#555" font-size="11" letter-spacing="3">HARDENING &amp; INFRASTRUCTURE</text>
<rect class="n" data-id="auth" x="30" y="728" width="250" height="38" rx="8" fill="#111827" stroke="#3b82f6" stroke-width="1.5"/>
<text x="155" y="752" text-anchor="middle" fill="#fff" font-size="12">API Key Auth + TTL Cache</text>
<rect class="n" data-id="ratelimit" x="330" y="728" width="250" height="38" rx="8" fill="#1a1033" stroke="#8b5cf6" stroke-width="1.5"/>
<text x="455" y="752" text-anchor="middle" fill="#fff" font-size="12">slowapi Rate Limiting</text>
<rect class="n" data-id="timeouts" x="630" y="728" width="250" height="38" rx="8" fill="#1a1505" stroke="#f59e0b" stroke-width="1.5"/>
<text x="755" y="752" text-anchor="middle" fill="#fff" font-size="12">Timeouts + Size Limits</text>
<rect class="n" data-id="responses" x="930" y="728" width="250" height="38" rx="8" fill="#051a14" stroke="#10b981" stroke-width="1.5"/>
<text x="1055" y="752" text-anchor="middle" fill="#fff" font-size="12">Standardized Responses</text>
<rect class="n" data-id="pagination" x="1230" y="728" width="260" height="38" rx="8" fill="#1a0a0a" stroke="#ef4444" stroke-width="1.5"/>
<text x="1360" y="752" text-anchor="middle" fill="#fff" font-size="12">Pagination (4 endpoints)</text>

<!-- ══ ROW 5: TESTS ══ -->
<text x="760" y="805" text-anchor="middle" fill="#555" font-size="11" letter-spacing="3">TEST COVERAGE</text>
<rect class="n" data-id="test-mvp" x="30" y="822" width="250" height="38" rx="8" fill="#0a1a0a" stroke="#22c55e" stroke-width="1.5"/>
<text x="155" y="846" text-anchor="middle" fill="#22c55e" font-size="12" font-weight="600">MVP: 108 tests</text>
<rect class="n" data-id="test-w1" x="330" y="822" width="250" height="38" rx="8" fill="#0a1a0a" stroke="#22c55e" stroke-width="1.5"/>
<text x="455" y="846" text-anchor="middle" fill="#22c55e" font-size="12" font-weight="600">Week 1: 85 tests</text>
<rect class="n" data-id="test-w2" x="630" y="822" width="250" height="38" rx="8" fill="#0a1a0a" stroke="#22c55e" stroke-width="1.5"/>
<text x="755" y="846" text-anchor="middle" fill="#22c55e" font-size="12" font-weight="600">Week 2: 68 tests</text>
<rect class="n" data-id="test-sim" x="930" y="822" width="250" height="38" rx="8" fill="#0a1a0a" stroke="#22c55e" stroke-width="1.5"/>
<text x="1055" y="846" text-anchor="middle" fill="#22c55e" font-size="12" font-weight="600">Chaos Sim: 37 tests</text>
<rect class="n" data-id="test-prod" x="1230" y="822" width="260" height="38" rx="8" fill="#0a1a0a" stroke="#22c55e" stroke-width="1.5"/>
<text x="1360" y="846" text-anchor="middle" fill="#22c55e" font-size="12" font-weight="600">Production: 46 tests</text>

<rect x="620" y="880" width="280" height="36" rx="18" fill="#22c55e" fill-opacity="0.12" stroke="#22c55e" stroke-width="1.5"/>
<text x="760" y="903" text-anchor="middle" fill="#22c55e" font-size="15" font-weight="bold">344 TESTS — ALL GREEN</text>

<!-- FEEDBACK LOOPS -->
<path d="M1360 580 L1360 690 Q1360 700 1345 700 L170 700 Q155 700 155 690 L155 375"
      fill="none" stroke="#3b82f6" stroke-width="1.5" stroke-dasharray="6,4" marker-end="url(#ab)"/>
<text x="760" y="697" text-anchor="middle" fill="#3b82f6" font-size="10" font-style="italic">model feedback: yolo11s-bar.pt auto-switches into DualTracker</text>

<path d="M1490 470 L1500 495 Q1500 505 1490 505 L310 505 Q300 505 300 515 L300 554 L328 554"
      fill="none" stroke="#10b981" stroke-width="1.5" stroke-dasharray="6,4" marker-end="url(#a)"/>
<text x="760" y="502" text-anchor="middle" fill="#10b981" font-size="10" font-style="italic">data feedback: confirmed_clips feed into training frames</text>
</svg>
</div>

<div id="tooltip" class="tt"></div>
<div class="dp"><div class="dc" id="dc"></div></div>

<script>
const D={
"video-src":{t:"Video Source",f:"app/video/download.py",d:"YouTube URLs, file uploads, or RTSP streams. yt-dlp with socket_timeout=30, retries=3."},
"download":{t:"yt-dlp Download",f:"app/video/download.py",d:"Downloads to temp. 720p max, MP4. Upload enforces MAX_UPLOAD_SIZE_MB."},
"pipeline":{t:"Video Pipeline",f:"app/video/pipeline.py",d:"Frame loop: detect, track, single-pass InsightFace (age+gender+embedding), behavior every 6th frame, store."},
"queue":{t:"Job Queue",f:"app/video/queue.py",d:"Background queue for batch/single video jobs. Status via GET /process/status/{job_id}."},
"yolo-mvp":{t:"YOLO11s",f:"app/video/models.py",d:"Person detection + BoT-SORT tracking. Conf 0.25. Singleton via get_yolo_model()."},
"insightface":{t:"InsightFace buffalo_l",f:"app/video/models.py",d:"Shared singleton. One app.get() call returns age, gender, 512-d embedding, quality. ~250MB."},
"yolo-pose":{t:"YOLO11s-Pose",f:"app/video/models.py",d:"17-keypoint pose. Classifies standing/sitting/walking/running/pointing/interacting."},
"face-embed":{t:"Face Embeddings",f:"app/video/embeddings.py",d:"512-dim ArcFace. Cosine similarity 0.45 for dedup. Concurrent cap prevents over-counting."},
"behavior":{t:"Behavior Analysis",f:"app/video/pipeline.py",d:"Pose classification updated every 6th processed frame. Six behavior classes."},
"demographics":{t:"Demographics",f:"app/video/helpers.py",d:"Age brackets (child/teen/young_adult/adult/senior) + gender from single-pass InsightFace."},
"fastapi":{t:"FastAPI Server",f:"app/__init__.py",d:"13 routers, 37 JSON endpoints, 8 HTML pages. X-API-Key auth, slowapi rate limits, response wrapper."},
"sqlite":{t:"SQLite + WAL",f:"app/database.py",d:"5 tables: venues, events, visitors, visitor_embeddings, alerts. Async via databases library."},
"html-ui":{t:"Web Dashboard",f:"app/routers/pages.py",d:"8 pages: Home, Analytics, Process, Dashboard, Analytics Dashboard, Uploads, Map, Architecture."},
"rtsp":{t:"RTSP / Video Feed",f:"sipp-pipeline/config/settings.py",d:"VIDEO_SOURCE env var. OpenCV VideoCapture. FPS auto-detected."},
"zones-cfg":{t:"Zone Config",f:"sipp-pipeline/config/zones.json",d:"Polygon zones per camera. Resolution-scaled Shapely. Point-in-polygon classification."},
"dual-tracker":{t:"DualTracker",f:"sipp-pipeline/pipeline/detector.py",d:"Two YOLO11s: BoT-SORT persons (occlusion), ByteTrack objects (motion). ~30 FPS A100."},
"interactions":{t:"InteractionDetector",f:"sipp-pipeline/pipeline/interactions.py",d:"IoU 0.1 + velocity 2px + 10s cooldown/pair + 100/hr budget. Blocks phantom triggers."},
"clip-buf":{t:"ClipBuffer",f:"sipp-pipeline/pipeline/clip_buffer.py",d:"Ring buffer 2x clip duration. Extracts 4 keyframes centered on event for VLM."},
"yolo-bar":{t:"YOLO11s / bar",f:"sipp-pipeline/config/settings.py",d:"Auto-switch: 'bar' in model name uses 17-class ontology (9 drinks). Else stock COCO (3)."},
"vlm":{t:"Claude Vision API",f:"sipp-pipeline/pipeline/verifier.py",d:"4 keyframes to Claude Sonnet. 9 actions: pouring_draft/bottle, mixing, serving, payment_card/cash, cleaning, idle, unknown."},
"safe-parse":{t:"Safe Parse",f:"sipp-pipeline/pipeline/verifier.py",d:"Strips fences, validates dict, ensures action+confidence. Handles 500s, arrays, nulls, binary."},
"dedup":{t:"EventDeduplicator",f:"sipp-pipeline/pipeline/deduplicator.py",d:"10s sliding window keyed zone:action. Keeps highest confidence. Purges expired."},
"cooldown":{t:"Cooldown + Velocity",f:"sipp-pipeline/pipeline/interactions.py",d:"10s cooldown per pair. Object must move 2px/frame (blocks stationary taps)."},
"budget":{t:"VLM Budget",f:"sipp-pipeline/pipeline/interactions.py",d:"100 calls/hour hard cap. Resets every 3600s."},
"bar-event":{t:"BarEvent",f:"sipp-pipeline/pipeline/events.py",d:"event_type, venue_id, camera_id, zone, confidence, action, drink/vessel, track IDs. Auto UUID."},
"event-api":{t:"POST /events/batch",f:"sipp-pipeline/pipeline/events.py",d:"Async httpx POST. 10s batch interval. Optional X-API-Key. 10s timeout."},
"clips":{t:"Training Clips",f:"sipp-pipeline/pipeline/clip_buffer.py",d:"confirmed_clips/{event_id}.mp4. Saved per verified event. Feeds training pipeline."},
"raw-video":{t:"Raw Footage",f:"sipp-pipeline/training/extract_frames.py",d:"Video files + confirmed_clips. --include-confirmed merges both sources."},
"extract":{t:"Frame Extraction",f:"sipp-pipeline/training/extract_frames.py",d:"ffmpeg at 1 FPS. JPGs to frames/ directory."},
"autolabel":{t:"Grounding DINO",f:"sipp-pipeline/training/autolabel.py",d:"Autodistill + CaptionOntology. 16 bar-specific prompts. Zero-shot labeling."},
"sahi":{t:"SAHI Sliced",f:"sipp-pipeline/training/sahi_label.py",d:"Small object detection at CCTV res. Low-confidence + 20% sample. NMS dedup."},
"remap":{t:"COCO Remap",f:"sipp-pipeline/training/remap_coco.py",d:"keyboard\u2192pos_terminal, dining_table\u2192bar_counter, chair\u2192bar_stool, tv\u2192bar_screen."},
"merge":{t:"Merge + NMS",f:"sipp-pipeline/training/merge_labels.py",d:"3 label sources unified. torchvision NMS dedup. Generates data.yaml."},
"train":{t:"YOLO Fine-tune",f:"sipp-pipeline/training/train.py",d:"50 epochs, 640px, auto batch. 80/10/10 split. Copies best.pt."},
"evaluate":{t:"Evaluate",f:"sipp-pipeline/training/evaluate.py",d:"Bar vs COCO side-by-side on test images. Per-class detection counts."},
"ontology":{t:"17-Class Ontology",f:"sipp-pipeline/training/config.py",d:"person, beer_glass, wine_glass, rocks_glass, shot_glass, cocktail_glass, pint_glass, beer_tap, liquor_bottle, beer_bottle, wine_bottle, pos_terminal, shaker, ice_bucket, bar_counter, bar_stool, bar_screen"},
"bar-model":{t:"yolo11s-bar.pt",f:"sipp-pipeline/training/train.py",d:"Fine-tuned YOLO11s. 17 classes. Set YOLO_MODEL=yolo11s-bar.pt for auto-switch."},
"data-yaml":{t:"data.yaml",f:"sipp-pipeline/training/merge_labels.py",d:"Training config: train/val/test paths + 17 class names."},
"auth":{t:"API Key Auth",f:"app/auth.py",d:"X-API-Key header lookup. TTLCache(1000, 300s). AUTH_ENABLED toggle. 33 endpoints."},
"ratelimit":{t:"Rate Limiting",f:"app/__init__.py",d:"slowapi: events 100/min, process 10/min, batch 10/min. By IP."},
"timeouts":{t:"Timeouts",f:"app/config.py",d:"yt-dlp socket_timeout=30. Upload Content-Length pre-check. Early 413."},
"responses":{t:"Response Wrapper",f:"app/responses.py",d:"{status, data, generated_at, pagination}. Health exempt."},
"pagination":{t:"Pagination",f:"app/routers/",d:"venues(50/200), alerts(50/200), batch(50/200), visitors(100/500)."},
"test-mvp":{t:"MVP Tests",f:"tests/",d:"108 tests: endpoints, edge cases, auth, rate limits, responses, pagination."},
"test-w1":{t:"Week 1 Tests",f:"sipp-pipeline/tests/",d:"85 tests: zones, interactions, dedup, verifier, torture."},
"test-w2":{t:"Week 2 Tests",f:"sipp-pipeline/training/tests/",d:"68 tests: merge, remap, SAHI, config, parse, NMS, roundtrips."},
"test-sim":{t:"Chaos Sim",f:"sipp-pipeline/tests/torture_sim_bar.py",d:"37 tests: 5000-frame sim, tracker drift, backpressure, VLM chaos, drops."},
"test-prod":{t:"Production Torture",f:"sipp-pipeline/tests/torture_production.py",d:"46 tests: 7 phases, 30 adversarial payloads, 10K marathon, security."}
};
const tt=document.getElementById('tooltip'),dc=document.getElementById('dc');
document.querySelectorAll('.n').forEach(n=>{
const id=n.dataset.id,d=D[id];if(!d)return;
n.style.cursor='pointer';
n.onmouseenter=()=>{tt.innerHTML='<div class="t1">'+d.t+'</div><div class="t2">'+d.f+'</div><div style="margin-top:6px">'+d.d+'</div>';tt.style.opacity='1';};
n.onmousemove=e=>{tt.style.left=(e.clientX+16)+'px';tt.style.top=(e.clientY+16)+'px';};
n.onmouseleave=()=>{tt.style.opacity='0';};
n.onclick=()=>{dc.className='dc a';dc.innerHTML='<h3>'+d.t+'</h3><p style="color:#3b82f6;font-family:monospace;font-size:12px">'+d.f+'</p><p style="margin-top:10px;color:#ccc;line-height:1.6">'+d.d+'</p>';dc.scrollIntoView({behavior:'smooth',block:'nearest'});};
});
</script>
</body>
</html>
"""

