"""
Behavior Analytics Endpoints (Phase 5)
=======================================
Engagement scores, behavior type breakdowns, posture analysis,
hourly engagement patterns, zone-level behavior, and printable HTML report.
"""

from datetime import datetime, timedelta

import sqlalchemy
from fastapi import APIRouter, Depends
from app.auth import require_api_key
from app.responses import success_response
from fastapi.responses import HTMLResponse

from app.database import database, events

router = APIRouter()


async def _behavior_analytics_data(venue_id: str, days: int = 7):
    """Internal helper: compute behavior analytics data as a dict."""
    cutoff = datetime.utcnow() - timedelta(days=days)

    query = """
        SELECT
            AVG(engagement_score) as avg_engagement,
            MIN(engagement_score) as min_engagement,
            MAX(engagement_score) as max_engagement,
            COUNT(CASE WHEN engagement_score >= 70 THEN 1 END) as highly_engaged,
            COUNT(CASE WHEN engagement_score >= 50 AND engagement_score < 70 THEN 1 END) as moderately_engaged,
            COUNT(CASE WHEN engagement_score < 50 THEN 1 END) as low_engaged,
            COUNT(engagement_score) as total_with_behavior
        FROM events
        WHERE venue_id = :venue_id
        AND timestamp >= :cutoff
        AND engagement_score IS NOT NULL
    """

    result = await database.fetch_one(query, {"venue_id": venue_id, "cutoff": cutoff})

    # Get behavior type breakdown
    behavior_query = """
        SELECT behavior_type, COUNT(*) as count
        FROM events
        WHERE venue_id = :venue_id
        AND timestamp >= :cutoff
        AND behavior_type IS NOT NULL
        GROUP BY behavior_type
        ORDER BY count DESC
    """
    behavior_rows = await database.fetch_all(behavior_query, {"venue_id": venue_id, "cutoff": cutoff})
    behavior_breakdown = {row["behavior_type"]: row["count"] for row in behavior_rows}

    # Get posture breakdown
    posture_query = """
        SELECT posture, COUNT(*) as count
        FROM events
        WHERE venue_id = :venue_id
        AND timestamp >= :cutoff
        AND posture IS NOT NULL
        GROUP BY posture
        ORDER BY count DESC
    """
    posture_rows = await database.fetch_all(posture_query, {"venue_id": venue_id, "cutoff": cutoff})
    posture_breakdown = {row["posture"]: row["count"] for row in posture_rows}

    # Get body orientation distribution
    orientation_query = """
        SELECT
            COUNT(CASE WHEN body_orientation > 0.3 THEN 1 END) as facing_camera,
            COUNT(CASE WHEN body_orientation <= 0.3 AND body_orientation >= -0.3 THEN 1 END) as sideways,
            COUNT(CASE WHEN body_orientation < -0.3 THEN 1 END) as facing_away,
            AVG(body_orientation) as avg_orientation
        FROM events
        WHERE venue_id = :venue_id
        AND timestamp >= :cutoff
        AND body_orientation IS NOT NULL
    """
    orientation = await database.fetch_one(orientation_query, {"venue_id": venue_id, "cutoff": cutoff})

    total = result["total_with_behavior"] or 0

    return {
        "venue_id": venue_id,
        "period_days": days,
        "engagement": {
            "average_score": round(result["avg_engagement"] or 0, 1),
            "min_score": round(result["min_engagement"] or 0, 1),
            "max_score": round(result["max_engagement"] or 0, 1),
            "highly_engaged_count": result["highly_engaged"] or 0,
            "highly_engaged_percent": round((result["highly_engaged"] or 0) / max(total, 1) * 100, 1),
            "moderately_engaged_count": result["moderately_engaged"] or 0,
            "low_engaged_count": result["low_engaged"] or 0,
        },
        "behavior_types": behavior_breakdown,
        "postures": posture_breakdown,
        "body_orientation": {
            "facing_camera": orientation["facing_camera"] or 0,
            "sideways": orientation["sideways"] or 0,
            "facing_away": orientation["facing_away"] or 0,
            "avg_orientation": round(orientation["avg_orientation"] or 0, 2),
        },
        "total_analyzed": total,
    }


@router.get("/analytics/{venue_id}/behavior")
async def get_behavior_analytics(venue_id: str, days: int = 7, _api_key: str = Depends(require_api_key)):
    """
    Get behavior and engagement analytics for a venue.
    Returns engagement scores, behavior type breakdown, and posture analysis.
    """
    return success_response(await _behavior_analytics_data(venue_id, days))


@router.get("/analytics/{venue_id}/behavior/hourly")
async def get_behavior_hourly(venue_id: str, days: int = 7, _api_key: str = Depends(require_api_key)):
    """
    Get hourly engagement patterns.
    Shows when visitors are most engaged throughout the day.
    """
    cutoff = datetime.utcnow() - timedelta(days=days)

    query = """
        SELECT
            CAST(strftime('%%H', timestamp) AS INTEGER) as hour,
            AVG(engagement_score) as avg_engagement,
            COUNT(*) as visitor_count,
            SUM(CASE WHEN behavior_type = 'engaged' THEN 1 ELSE 0 END) as engaged_count,
            SUM(CASE WHEN behavior_type = 'browsing' THEN 1 ELSE 0 END) as browsing_count,
            SUM(CASE WHEN behavior_type = 'passing' THEN 1 ELSE 0 END) as passing_count
        FROM events
        WHERE venue_id = :venue_id
        AND timestamp >= :cutoff
        AND engagement_score IS NOT NULL
        GROUP BY hour
        ORDER BY hour
    """

    rows = await database.fetch_all(query, {"venue_id": venue_id, "cutoff": cutoff})

    hourly_data = []
    peak_engagement_hour = 0
    peak_engagement_score = 0

    for row in rows:
        avg_eng = row["avg_engagement"] or 0
        if avg_eng > peak_engagement_score:
            peak_engagement_score = avg_eng
            peak_engagement_hour = row["hour"]

        hourly_data.append({
            "hour": row["hour"],
            "avg_engagement": round(avg_eng, 1),
            "visitor_count": row["visitor_count"],
            "engaged_count": row["engaged_count"] or 0,
            "browsing_count": row["browsing_count"] or 0,
            "passing_count": row["passing_count"] or 0,
        })

    return success_response({
        "venue_id": venue_id,
        "period_days": days,
        "hourly_engagement": hourly_data,
        "peak_engagement_hour": peak_engagement_hour,
        "peak_engagement_score": round(peak_engagement_score, 1),
        "insight": f"Visitors are most engaged at {peak_engagement_hour}:00 (avg score: {round(peak_engagement_score, 1)})"
    })


@router.get("/analytics/{venue_id}/behavior/zones")
async def get_behavior_by_zone(venue_id: str, days: int = 7, _api_key: str = Depends(require_api_key)):
    """
    Get engagement metrics by zone.
    Shows which areas of the venue have the most engaged visitors.
    """
    cutoff = datetime.utcnow() - timedelta(days=days)

    query = """
        SELECT
            zone,
            AVG(engagement_score) as avg_engagement,
            COUNT(*) as visitor_count,
            AVG(dwell_seconds) as avg_dwell,
            SUM(CASE WHEN behavior_type = 'engaged' THEN 1 ELSE 0 END) as engaged_count,
            SUM(CASE WHEN behavior_type = 'browsing' THEN 1 ELSE 0 END) as browsing_count,
            SUM(CASE WHEN behavior_type = 'waiting' THEN 1 ELSE 0 END) as waiting_count,
            SUM(CASE WHEN behavior_type = 'passing' THEN 1 ELSE 0 END) as passing_count
        FROM events
        WHERE venue_id = :venue_id
        AND timestamp >= :cutoff
        AND engagement_score IS NOT NULL
        GROUP BY zone
        ORDER BY avg_engagement DESC
    """

    rows = await database.fetch_all(query, {"venue_id": venue_id, "cutoff": cutoff})

    zones = []
    for row in rows:
        total = row["visitor_count"] or 1
        zones.append({
            "zone": row["zone"],
            "avg_engagement": round(row["avg_engagement"] or 0, 1),
            "visitor_count": row["visitor_count"],
            "avg_dwell_seconds": round(row["avg_dwell"] or 0, 1),
            "behavior_mix": {
                "engaged": round((row["engaged_count"] or 0) / total * 100, 1),
                "browsing": round((row["browsing_count"] or 0) / total * 100, 1),
                "waiting": round((row["waiting_count"] or 0) / total * 100, 1),
                "passing": round((row["passing_count"] or 0) / total * 100, 1),
            }
        })

    # Find best and worst zones
    best_zone = zones[0]["zone"] if zones else "unknown"
    worst_zone = zones[-1]["zone"] if zones else "unknown"

    return success_response({
        "venue_id": venue_id,
        "period_days": days,
        "zones": zones,
        "insights": {
            "highest_engagement_zone": best_zone,
            "lowest_engagement_zone": worst_zone,
            "recommendation": f"Focus on improving engagement in the '{worst_zone}' zone" if worst_zone != best_zone else "All zones performing similarly"
        }
    })


@router.get("/report/{venue_id}", response_class=HTMLResponse)
async def generate_report(venue_id: str, days: int = 7):
    """
    Generate printable HTML report (Nielsen-style).
    Open in browser and print to PDF.
    """
    import html
    venue_id = html.escape(venue_id)
    from app.routers.advanced_analytics import _executive_summary_data, _demographics_data, _zone_data

    # Gather data (use internal helpers to get raw dicts)
    summary = await _executive_summary_data(venue_id, days)
    demographics = await _demographics_data(venue_id, days)
    zones = await _zone_data(venue_id, days)

    current = summary["current"]
    change = summary["change"]

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Analytics Report - {venue_id}</title>
        <style>
            @media print {{
                body {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
                .no-print {{ display: none; }}
            }}
            * {{ box-sizing: border-box; margin: 0; padding: 0; }}
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #fff;
                color: #1a1a1a;
                padding: 40px;
                max-width: 900px;
                margin: 0 auto;
            }}
            .header {{
                border-bottom: 3px solid #3b82f6;
                padding-bottom: 20px;
                margin-bottom: 30px;
            }}
            .header h1 {{ font-size: 28px; margin-bottom: 5px; }}
            .header .subtitle {{ color: #666; }}
            .header .period {{ color: #3b82f6; font-weight: 600; margin-top: 10px; }}

            .section {{ margin-bottom: 40px; }}
            .section h2 {{
                font-size: 18px;
                color: #333;
                border-bottom: 1px solid #ddd;
                padding-bottom: 10px;
                margin-bottom: 20px;
            }}

            .kpi-grid {{
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 20px;
            }}
            .kpi-card {{
                background: #f8f9fa;
                border-radius: 8px;
                padding: 20px;
                text-align: center;
            }}
            .kpi-value {{ font-size: 32px; font-weight: bold; color: #1a1a1a; }}
            .kpi-label {{ color: #666; font-size: 12px; margin-top: 5px; text-transform: uppercase; }}
            .kpi-change {{ font-size: 12px; margin-top: 5px; }}
            .kpi-change.positive {{ color: #22c55e; }}
            .kpi-change.negative {{ color: #ef4444; }}

            .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 30px; }}

            .demo-bars {{ }}
            .demo-bar {{
                display: flex;
                align-items: center;
                margin-bottom: 10px;
            }}
            .demo-label {{ width: 80px; font-size: 14px; color: #666; }}
            .demo-track {{
                flex: 1;
                height: 24px;
                background: #e5e7eb;
                border-radius: 4px;
                overflow: hidden;
            }}
            .demo-fill {{
                height: 100%;
                background: #3b82f6;
                display: flex;
                align-items: center;
                justify-content: flex-end;
                padding-right: 8px;
                color: white;
                font-size: 12px;
                font-weight: 600;
            }}
            .demo-fill.female {{ background: #ec4899; }}

            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #e5e7eb;
            }}
            th {{ background: #f8f9fa; font-weight: 600; color: #666; font-size: 12px; text-transform: uppercase; }}

            .insights {{
                background: #eff6ff;
                border-left: 4px solid #3b82f6;
                padding: 20px;
                border-radius: 0 8px 8px 0;
            }}
            .insights h3 {{ margin-bottom: 15px; color: #1e40af; }}
            .insights ul {{ list-style: none; }}
            .insights li {{ padding: 8px 0; padding-left: 20px; position: relative; }}
            .insights li:before {{ content: "->"; position: absolute; left: 0; color: #3b82f6; }}

            .footer {{
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
                color: #888;
                font-size: 12px;
                text-align: center;
            }}

            .print-btn {{
                position: fixed;
                top: 20px;
                right: 20px;
                background: #3b82f6;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                cursor: pointer;
                font-weight: 600;
            }}
            .print-btn:hover {{ background: #2563eb; }}
        </style>
    </head>
    <body>
        <button class="print-btn no-print" onclick="window.print()">Print / Save PDF</button>

        <div class="header">
            <h1>Analytics Report</h1>
            <div class="subtitle">Venue: {venue_id}</div>
            <div class="period">Period: Last {days} days - Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</div>
        </div>

        <div class="section">
            <h2>Key Performance Indicators</h2>
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-value">{current['unique_visitors']}</div>
                    <div class="kpi-label">Unique Visitors</div>
                    <div class="kpi-change {'positive' if change['visitors'] >= 0 else 'negative'}">
                        {'+' if change['visitors'] >= 0 else ''}{change['visitors']} ({'+' if change['visitors_percent'] >= 0 else ''}{change['visitors_percent']}%)
                    </div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-value">{current['return_rate_percent']}%</div>
                    <div class="kpi-label">Return Rate</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-value">{current['avg_dwell_minutes']}</div>
                    <div class="kpi-label">Avg Dwell (min)</div>
                    <div class="kpi-change {'positive' if change['dwell_minutes'] >= 0 else 'negative'}">
                        {'+' if change['dwell_minutes'] >= 0 else ''}{change['dwell_minutes']} min
                    </div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-value">{current['peak_hour'] or '-'}:00</div>
                    <div class="kpi-label">Peak Hour</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Demographics</h2>
            <div class="two-col">
                <div>
                    <h4 style="margin-bottom: 15px; color: #666;">Gender Distribution</h4>
                    <div class="demo-bars">
                        <div class="demo-bar">
                            <span class="demo-label">Male</span>
                            <div class="demo-track">
                                <div class="demo-fill" style="width: {demographics['current']['gender'].get('M', 0)}%">
                                    {demographics['current']['gender'].get('M', 0)}%
                                </div>
                            </div>
                        </div>
                        <div class="demo-bar">
                            <span class="demo-label">Female</span>
                            <div class="demo-track">
                                <div class="demo-fill female" style="width: {demographics['current']['gender'].get('F', 0)}%">
                                    {demographics['current']['gender'].get('F', 0)}%
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                <div>
                    <h4 style="margin-bottom: 15px; color: #666;">Age Distribution</h4>
                    <div class="demo-bars">
                        {''.join(f"""
                        <div class="demo-bar">
                            <span class="demo-label">{age}</span>
                            <div class="demo-track">
                                <div class="demo-fill" style="width: {pct}%; background: {'#3b82f6' if age == '20s' else '#8b5cf6' if age == '30s' else '#f59e0b' if age == '40s' else '#ef4444'}">
                                    {pct}%
                                </div>
                            </div>
                        </div>
                        """ for age, pct in demographics['current']['age'].items())}
                    </div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Zone Performance</h2>
            <table>
                <thead>
                    <tr>
                        <th>Zone</th>
                        <th>Visitors</th>
                        <th>Traffic %</th>
                        <th>Avg Dwell</th>
                        <th>Engagement</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(f"""
                    <tr>
                        <td>{z['zone']}</td>
                        <td>{z['visitors']}</td>
                        <td>{z['traffic_percent']}%</td>
                        <td>{z['avg_dwell_minutes']} min</td>
                        <td>{z['engagement']}</td>
                    </tr>
                    """ for z in zones['zones'][:10])}
                </tbody>
            </table>
        </div>

        <div class="section">
            <div class="insights">
                <h3>Key Insights</h3>
                <ul>
                    {''.join(f'<li>{insight}</li>' for insight in summary['insights']) if summary['insights'] else '<li>Process more videos to generate insights</li>'}
                </ul>
            </div>
        </div>

        <div class="footer">
            Generated by CCTV Analytics Platform - {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
        </div>
    </body>
    </html>
    """
