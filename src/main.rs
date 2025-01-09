use dotenv::dotenv;
use std::env;
use reqwest;
use serde::Deserialize;

use pyo3::prelude::*;
use pyo3::types::IntoPyDict;

use eframe::egui;
use std::sync::{Arc, Mutex};

fn string_to_f64<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s: String = Deserialize::deserialize(deserializer)?;
    s.parse::<f64>().map_err(serde::de::Error::custom)
}

#[derive(Debug, Deserialize)]
pub struct InflationData {
    date: String,
    #[serde(deserialize_with = "string_to_f64")]
    value: f64
}

#[derive(Debug, Deserialize)]
pub struct InflationRaw {
    inflation: Vec<InflationData>
}

struct RustFin {
    country: String,
}

pub struct HistoricalSeriesDates {
    pub date: String,
}

pub struct HistoricalSeriesValues {
    pub value: f64,
}

pub async fn get_historical_inflation(country: &str) -> Result<InflationRaw, Box<dyn std::error::Error>> {
    dotenv().ok();
    let token = env::var("API_TOKEN").expect("API_TOKEN not found");
    let url_base = env::var("URL_BASE").expect("URL_BASE not found");

    let url = format!(
        "{}inflation?country={}&historical=true&sortBy=date&sortOrder=desc&token={}",
        url_base, country, token
    );

    let response = reqwest::get(&url).await?;        
    
    let data: InflationRaw = response.json().await?;

    Ok(data)
}

impl RustFin {
    fn new(country: &str) -> Self {
        Self {
            country: country.to_string(),
        }
    }

    async fn make_historical_array(&self) -> Result<Vec<InflationData>, Box<dyn std::error::Error>> {
        let inflation_raw = get_historical_inflation(&self.country).await?;

        Ok(inflation_raw.inflation)
    }
}

async fn get_historical_data(
    country: &str,
) -> Result<Vec<HistoricalSeriesValues>, Box<dyn std::error::Error>> {
    let rust_fin = RustFin::new(country);
    let inflation_data = rust_fin.make_historical_array().await?;

    let mut values: Vec<HistoricalSeriesValues> = Vec::new();
    let mut dates: Vec<HistoricalSeriesDates> = Vec::new();

    for item in inflation_data {
        let date = HistoricalSeriesDates {
            date: item.date.to_string(),
        };

        let value = HistoricalSeriesValues {
            value: item.value,
        };

        values.push(value);
        dates.push(date);
    }

    Ok(values)
}


fn run_arima_model(values: &[f64], p: u32, d: u32, q_arg: u32) -> PyResult<Vec<f64>> {
    Python::with_gil(|py| {
        let statsmodels = py
            .import("statsmodels.tsa.arima.model")
            .expect("Erro ao importar statsmodels.tsa.arima.model");
        let numpy = py.import("numpy").expect("Erro ao importar numpy");        

        // Convertendo valores para array numpy
        let np_array = numpy
            .call_method1("array", (values.to_vec(),))
            .expect("Erro ao criar o array numpy");

        // Criando o dicionário de parâmetros
        let kwargs = [("order", (p, d, q_arg))].into_py_dict(py);
        let arima_model = statsmodels
            .call_method("ARIMA", (np_array,), Some(kwargs))?
            .call_method0("fit")?;

        // Fazendo previsões (5 passos futuros)
        let forecast = arima_model.call_method1("forecast", (150,))?;
        let forecast_values: Vec<f64> = forecast.extract()?;
        Ok(forecast_values)
    })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();
    pyo3::prepare_freethreaded_python();

    let app = MyApp::new().await;

    // Inicializa a interface gráfica
    eframe::run_native(
        "ARIMA Model Visualization",
        eframe::NativeOptions::default(),
        Box::new(|_cc| Box::new(app)),
    )
    .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;

    Ok(())
}

pub struct MyApp {
    historical: Arc<Mutex<Vec<f64>>>,
    forecast: Arc<Mutex<Vec<f64>>>,    
    values: Arc<Mutex<Vec<f64>>>,
    predictions: Arc<Mutex<Vec<f64>>>,    
    p: u32,
    d: u32,
    q: u32,
}

impl MyApp {
    pub async fn new() -> Self {        
        let q_country = "brazil";        
        let values = get_historical_data(q_country).await.unwrap();

        let values: Vec<f64> = values.iter().map(|v| v.value).collect();

        let predictions = Arc::new(Mutex::new(vec![]));
        let historical = Arc::new(Mutex::new(values.clone()));
        let forecast = Arc::new(Mutex::new(vec![]));

        Self {
            historical,
            forecast,                       
            values: Arc::new(Mutex::new(values)),
            predictions,            
            p: 1,
            d: 1,
            q: 1,
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("ARIMA Model Visualization");            
            // Ajuste dos parâmetros
            ui.add(egui::Slider::new(&mut self.p, 0..=10).text("p (AR) (Representa o número de termos passados da série que serão usados para prever o próximo valor)"));
            ui.add(egui::Slider::new(&mut self.d, 0..=10).text("d (I) (Representa o número de diferenças que serão aplicadas na série para torná-la estacionária (sem tendência ou sazonalidade))"));
            ui.add(egui::Slider::new(&mut self.q, 0..=10).text("q (MA) (representa o número de erros passados que serão usados para ajustar a previsão atual)"));

            if ui.button("Recalcular Previsões").clicked() {
                // Recalcular previsões ao clicar
                let values = self.values.lock().unwrap().clone();                

                let forecast = run_arima_model(&values, self.p, self.d, self.q).unwrap_or_else(|_| vec![]);
                *self.predictions.lock().unwrap() = forecast.clone();                
            }                        
            
            // Exibição de previsões
            ui.label("Previsões:");
            egui::ScrollArea::vertical().show(ui, |ui| {
                for (i, forecast) in self.predictions.lock().unwrap().iter().enumerate() {
                    ui.label(format!("Passo {}: {:.2}", i + 1, forecast));
                }
            });
        });

        ctx.request_repaint(); // Atualiza continuamente a interface
    }
}
