import express, { Request, Response } from 'express';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import { createServer as createViteServer } from 'vite';
import { exec } from 'child_process';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;
const isProd = process.env.NODE_ENV === 'production';

app.use(express.json());

// API: Get current verification metrics and model comparisons
app.get('/api/metrics', (req: Request, res: Response) => {
  const metricsPath = path.join(__dirname, 'results', 'summary_metrics.json');
  if (fs.existsSync(metricsPath)) {
    try {
      const data = JSON.parse(fs.readFileSync(metricsPath, 'utf-8'));
      return res.json({ success: true, data });
    } catch (e: any) {
      return res.status(500).json({ success: false, error: e.message });
    }
  }
  return res.json({ success: false, message: 'Summary metrics not generated yet. Trigger pipeline or wait.' });
});

// API: Get District-level forecasts
app.get('/api/districts', (req: Request, res: Response) => {
  const metricsPath = path.join(__dirname, 'results', 'summary_metrics.json');
  if (fs.existsSync(metricsPath)) {
    try {
      const data = JSON.parse(fs.readFileSync(metricsPath, 'utf-8'));
      return res.json({ success: true, districts: data.district_forecasts || [] });
    } catch (e: any) {
      return res.status(500).json({ success: false, error: e.message });
    }
  }
  return res.json({ success: false, districts: [] });
});

// API: Download District Forecasts CSV
app.get(['/api/export/districts-csv', '/api/export/districts.csv', '/api/districts/csv'], (req: Request, res: Response) => {
  const csvDiskPath = path.join(__dirname, 'results', 'district_forecasts.csv');
  if (fs.existsSync(csvDiskPath)) {
    res.setHeader('Content-Type', 'text/csv; charset=utf-8');
    res.setHeader('Content-Disposition', 'attachment; filename="india_monsoon_district_forecasts.csv"');
    return res.sendFile(csvDiskPath);
  }

  const metricsPath = path.join(__dirname, 'results', 'summary_metrics.json');
  if (fs.existsSync(metricsPath)) {
    try {
      const data = JSON.parse(fs.readFileSync(metricsPath, 'utf-8'));
      const districts = data.district_forecasts || [];
      
      const headers = [
        'District',
        'State',
        'Zone',
        'Regime',
        'Raw NWP Max (mm)',
        'AI Corrected Max (mm)',
        'Delta Correction (mm)',
        'Observed Mean (mm)',
        'P(Heavy >=64.5mm)',
        'P(Very Heavy >=115.6mm)',
        'P(Extreme >=204.5mm)',
        'Rainfall Category'
      ];

      const rows = districts.map((d: any) => [
        `"${String(d.district || '').replace(/"/g, '""')}"`,
        `"${String(d.state || '').replace(/"/g, '""')}"`,
        `"${String(d.zone || '').replace(/"/g, '""')}"`,
        `"${String(d.regime || '').replace(/"/g, '""')}"`,
        d.raw_nwp_max ?? 0,
        d.corrected_max ?? 0,
        d.delta_correction ?? 0,
        d.observed_mean ?? 0,
        `${((d.p_heavy ?? 0) * 100).toFixed(1)}%`,
        `${((d.p_very_heavy ?? 0) * 100).toFixed(1)}%`,
        `${((d.p_extreme ?? 0) * 100).toFixed(1)}%`,
        `"${String(d.category || '').replace(/"/g, '""')}"`
      ]);

      const csvContent = '\uFEFF' + [headers.join(','), ...rows.map((r: any) => r.join(','))].join('\n');
      res.setHeader('Content-Type', 'text/csv; charset=utf-8');
      res.setHeader('Content-Disposition', 'attachment; filename="india_monsoon_district_forecasts.csv"');
      return res.send(csvContent);
    } catch (e: any) {
      return res.status(500).json({ success: false, error: e.message });
    }
  }

  return res.status(404).send('District forecasts data not found. Please run the pipeline first.');
});

// API: Download Model Comparison & Verification CSV
app.get(['/api/export/verification-csv', '/api/export/verification.csv', '/api/verification/csv', '/api/export/model-comparison.csv'], (req: Request, res: Response) => {
  const csvDiskPath = path.join(__dirname, 'results', 'model_comparison.csv');
  if (fs.existsSync(csvDiskPath)) {
    res.setHeader('Content-Type', 'text/csv; charset=utf-8');
    res.setHeader('Content-Disposition', 'attachment; filename="model_comparison_verification.csv"');
    return res.sendFile(csvDiskPath);
  }

  const metricsPath = path.join(__dirname, 'results', 'summary_metrics.json');
  if (fs.existsSync(metricsPath)) {
    try {
      const data = JSON.parse(fs.readFileSync(metricsPath, 'utf-8'));
      const comparisonTable = data.model_comparison?.comparison_table || [];
      
      const headers = [
        'Model Architecture',
        'RMSE (mm)',
        'MAE (mm)',
        'Bias (mm)',
        'CSI (Threat Score)',
        'ETS (Equitable Threat)',
        'POD (Probability of Detection)',
        'FAR (False Alarm Ratio)',
        'Frequency Bias',
        'FSS (Fractions Skill Score)',
        'Hits',
        'Misses',
        'False Alarms'
      ];

      const rows = comparisonTable.map((m: any) => [
        `"${String(m.model || '').replace(/"/g, '""')}"`,
        m.rmse ?? '',
        m.mae ?? '',
        m.bias ?? '',
        m.csi ?? '',
        m.ets ?? '',
        m.pod ?? '',
        m.far ?? '',
        m.frequency_bias ?? '',
        m.fss ?? '',
        m.hits ?? '',
        m.misses ?? '',
        m.false_alarms ?? ''
      ]);

      const regimeBreakdown = data.regime_wise_verification?.regime_breakdown || [];
      const regimeRows: string[] = [];
      if (regimeBreakdown.length > 0) {
        regimeRows.push('\n# REGIME-WISE VERIFICATION BREAKDOWN');
        regimeRows.push('Regime,Event Count,Observed Mean (mm),Model,RMSE (mm),MAE (mm),Bias (mm),CSI,POD,FAR');
        for (const r of regimeBreakdown) {
          const regName = `"${r.regime}"`;
          const count = r.count;
          const obsMean = r.obs_mean;
          for (const [modelName, stats] of Object.entries<any>(r.models || {})) {
            regimeRows.push([
              regName,
              count,
              obsMean,
              `"${modelName}"`,
              stats.rmse ?? '',
              stats.mae ?? '',
              stats.bias ?? '',
              stats.csi ?? '',
              stats.pod ?? '',
              stats.far ?? ''
            ].join(','));
          }
        }
      }

      const csvContent = '\uFEFF' + [headers.join(','), ...rows.map((r: any) => r.join(',')), ...regimeRows].join('\n');
      res.setHeader('Content-Type', 'text/csv; charset=utf-8');
      res.setHeader('Content-Disposition', 'attachment; filename="model_comparison_verification.csv"');
      return res.send(csvContent);
    } catch (e: any) {
      return res.status(500).json({ success: false, error: e.message });
    }
  }

  return res.status(404).send('Verification data not found. Please run the pipeline first.');
});

// API: Real-time inference endpoint for custom meteorological inputs
app.post('/api/predict', (req: Request, res: Response) => {
  try {
    const {
      latitude = 18.96,
      longitude = 72.82,
      rainfall = 82.0,
      humidity = 88.0,
      temperature = 26.0,
      wind_speed = 12.0,
      elevation = 14.0,
      coast_dist_km = 2.0,
      pressure = 998.0,
      cape = 2100.0,
      vertical_velocity = -0.35,
      district_name = 'Custom Station'
    } = req.body;

    const lat = Number(latitude);
    const lon = Number(longitude);
    const rain = Math.max(0, Number(rainfall));
    const rh = Math.min(100, Math.max(0, Number(humidity)));
    const wind = Math.max(0, Number(wind_speed));
    const elev = Math.max(0, Number(elevation));
    const coast = Math.max(0, Number(coast_dist_km));
    const pres = Number(pressure);
    const cap = Math.max(0, Number(cape));
    const omega = Number(vertical_velocity);

    // Weather regime classification logic
    let regime = 'normal_monsoon';
    if (rain >= 100.0 || (rain >= 60.0 && cap > 2400 && omega < -0.4)) {
      regime = 'extreme_event';
    } else if (pres <= 998.0 && wind >= 12.0 && rain >= 35.0) {
      regime = 'monsoon_depression';
    } else if ((elev >= 450.0 || (lat < 20.0 && lon < 76.5 && elev >= 200.0)) && rain >= 25.0 && rh >= 80.0) {
      regime = 'orographic_rainfall';
    } else if (coast <= 35.0 && rh >= 82.0 && rain >= 15.0) {
      regime = 'coastal_rainfall';
    } else if (lat >= 28.0 && pres <= 1004.0 && (wind >= 9.0 || elev > 600) && rain >= 10.0) {
      regime = 'western_disturbance';
    } else if (rain >= 25.0 && rh >= 80.0 && wind >= 8.0) {
      regime = 'active_monsoon';
    } else if (rain < 5.0 && rh < 68.0 && pres >= 1008.0) {
      regime = 'break_monsoon';
    }

    // Regime-aware AI bias correction delta
    let corrected = rain;
    const factors: any[] = [];

    if (regime === 'orographic_rainfall') {
      const orographicLift = Math.min(35, (elev / 500.0) * 12.0 + (rh / 100.0) * 8.0);
      corrected = rain * 1.35 + orographicLift;
      factors.push({
        name: 'Orographic Moisture Trapping',
        impact: `+${orographicLift.toFixed(1)} mm`,
        detail: `Steep Western Ghats terrain (${elev}m) forces vigorous mechanical ascent not resolved by coarse NWP.`
      });
    } else if (regime === 'active_monsoon') {
      const activeEnhance = rain * 0.22 + 6.5;
      corrected = rain + activeEnhance;
      factors.push({
        name: 'Active Monsoon Convective Underestimation',
        impact: `+${activeEnhance.toFixed(1)} mm`,
        detail: 'Synoptic monsoon trough active conditions historically exhibit NWP convective dry bias.'
      });
    } else if (regime === 'monsoon_depression') {
      const deprEnhance = rain * 0.28 + (1004 - pres) * 1.5;
      corrected = rain + deprEnhance;
      factors.push({
        name: 'Depression Vortex Intensity Correction',
        impact: `+${deprEnhance.toFixed(1)} mm`,
        detail: `Low pressure anomaly (${pres} hPa) drives cyclonic convergence with moisture influx.`
      });
    } else if (regime === 'coastal_rainfall') {
      const coastDelta = rain * 0.18 + 4.0;
      corrected = rain + coastDelta;
      factors.push({
        name: 'Coastal Boundary Convergence',
        impact: `+${coastDelta.toFixed(1)} mm`,
        detail: `Offshore moisture convergence near coastline (${coast} km) enhances precipitable water.`
      });
    } else if (regime === 'break_monsoon') {
      const reduction = Math.min(rain * 0.45, 12.0);
      corrected = Math.max(0, rain - reduction);
      factors.push({
        name: 'Break Monsoon Suppression',
        impact: `-${reduction.toFixed(1)} mm`,
        detail: 'Suppressed convection and dry air intrusion dampen spurious NWP false-alarm rainfall.'
      });
    } else {
      corrected = rain * 1.05;
      factors.push({
        name: 'Climatological Background Calibration',
        impact: '+5.0%',
        detail: 'Slight positive calibration based on regional seasonal verification stats.'
      });
    }

    corrected = Math.max(0, Math.min(1500, Math.round(corrected * 10) / 10));

    // Calibrated heavy rainfall probabilities (sigmoid around thresholds)
    const pHeavy = Math.min(0.99, Math.max(0.01, 1.0 / (1.0 + Math.exp(-(corrected - 64.5) / 14.0))));
    const pVeryHeavy = Math.min(0.98, Math.max(0.01, 1.0 / (1.0 + Math.exp(-(corrected - 115.6) / 22.0))));
    const pExtreme = Math.min(0.95, Math.max(0.01, 1.0 / (1.0 + Math.exp(-(corrected - 204.5) / 35.0))));

    return res.json({
      success: true,
      prediction: {
        district: district_name,
        regime,
        raw_rainfall: rain,
        corrected_rainfall: corrected,
        delta: Math.round((corrected - rain) * 10) / 10,
        heavy_probability: Math.round(pHeavy * 1000) / 10,
        very_heavy_probability: Math.round(pVeryHeavy * 1000) / 10,
        extreme_probability: Math.round(pExtreme * 1000) / 10,
        explainability_factors: factors,
        timestamp: new Date().toISOString()
      }
    });
  } catch (err: any) {
    return res.status(500).json({ success: false, error: err.message });
  }
});

// API: Re-run the python pipeline
app.post('/api/pipeline/run', (req: Request, res: Response) => {
  exec('python3 run.py demo', (error, stdout, stderr) => {
    if (error) {
      console.error(`Pipeline error: ${stderr}`);
      return res.status(500).json({ success: false, error: stderr || error.message });
    }
    const metricsPath = path.join(__dirname, 'results', 'summary_metrics.json');
    if (fs.existsSync(metricsPath)) {
      const data = JSON.parse(fs.readFileSync(metricsPath, 'utf-8'));
      return res.json({ success: true, message: 'Pipeline executed successfully', data });
    }
    return res.json({ success: true, output: stdout });
  });
});

async function startServer() {
  if (!isProd) {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.join(__dirname, 'dist')));
    app.get('*', (req: Request, res: Response) => {
      res.sendFile(path.join(__dirname, 'dist', 'index.html'));
    });
  }

  app.listen(PORT, () => {
    console.log(`Server running on port ${PORT} in ${isProd ? 'production' : 'development'} mode`);
  });
}

startServer();
