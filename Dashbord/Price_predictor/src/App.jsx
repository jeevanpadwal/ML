import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const PricePredictionDashboard = () => {
  // Sample data - in production this would come from your model API
  const [predictionData, setPredictionData] = useState([
    { date: '2025-01-01', actualPrice: 45.20, predictedPrice: 44.80, t2m: 22.5, cloudAmt: 0.3, windSpeed: 4.2 },
    { date: '2025-01-02', actualPrice: 44.90, predictedPrice: 45.10, t2m: 23.1, cloudAmt: 0.4, windSpeed: 3.8 },
    { date: '2025-01-03', actualPrice: 46.50, predictedPrice: 46.20, t2m: 23.8, cloudAmt: 0.2, windSpeed: 3.5 },
    { date: '2025-01-04', actualPrice: 47.80, predictedPrice: 47.50, t2m: 24.2, cloudAmt: 0.1, windSpeed: 3.2 },
    { date: '2025-01-05', actualPrice: 48.20, predictedPrice: 48.40, t2m: 24.5, cloudAmt: 0.1, windSpeed: 3.0 },
    { date: '2025-01-06', actualPrice: 47.90, predictedPrice: 48.10, t2m: 24.8, cloudAmt: 0.2, windSpeed: 3.1 },
    { date: '2025-01-07', actualPrice: 49.10, predictedPrice: 48.50, t2m: 25.1, cloudAmt: 0.3, windSpeed: 3.4 },
    { date: '2025-01-08', actualPrice: 50.30, predictedPrice: 49.80, t2m: 25.3, cloudAmt: 0.2, windSpeed: 3.6 },
    { date: '2025-01-09', actualPrice: 51.20, predictedPrice: 50.70, t2m: 25.5, cloudAmt: 0.1, windSpeed: 3.8 },
    { date: '2025-01-10', actualPrice: 52.40, predictedPrice: 51.90, t2m: 25.8, cloudAmt: 0.0, windSpeed: 4.0 },
    { date: '2025-01-11', actualPrice: 53.10, predictedPrice: 52.80, t2m: 26.0, cloudAmt: 0.0, windSpeed: 4.2 },
    { date: '2025-01-12', actualPrice: 54.30, predictedPrice: 53.90, t2m: 26.2, cloudAmt: 0.1, windSpeed: 4.5 },
    { date: '2025-01-13', actualPrice: 55.20, predictedPrice: 54.80, t2m: 26.5, cloudAmt: 0.2, windSpeed: 4.8 },
    { date: '2025-01-14', actualPrice: 54.80, predictedPrice: 55.10, t2m: 26.8, cloudAmt: 0.3, windSpeed: 5.0 },
  ]);

  const [selectedFeature, setSelectedFeature] = useState('t2m');
  const [forecastDays, setForecastDays] = useState(7);
  const [commodity, setCommodity] = useState('Tomato');

  // Calculate metrics
  const calculateMetrics = () => {
    let mse = 0;
    let mae = 0;
    let count = 0;
    
    predictionData.forEach(data => {
      if (data.actualPrice && data.predictedPrice) {
        mse += Math.pow(data.actualPrice - data.predictedPrice, 2);
        mae += Math.abs(data.actualPrice - data.predictedPrice);
        count++;
      }
    });
    
    return {
      mse: (mse / count).toFixed(4),
      mae: (mae / count).toFixed(2),
      rmse: Math.sqrt(mse / count).toFixed(2)
    };
  };
  
  const metrics = calculateMetrics();

  return (
    <div className="flex flex-col p-6 bg-gray-50 w-full">
      <div className="bg-white rounded-lg shadow p-6 mb-6">
        <h1 className="text-2xl font-bold text-gray-800 mb-2">Agricultural Price Prediction Dashboard</h1>
        <p className="text-gray-600">Monitor and forecast commodity prices based on weather conditions</p>
        
        <div className="flex flex-wrap gap-4 mt-4">
          <div className="flex items-center">
            <label className="mr-2 text-gray-700">Commodity:</label>
            <select 
              className="border rounded p-2 bg-white"
              value={commodity}
              onChange={(e) => setCommodity(e.target.value)}
            >
              <option value="Tomato">Tomato</option>
              <option value="Potato">Potato</option>
              <option value="Onion">Onion</option>
              <option value="Rice">Rice</option>
              <option value="Wheat">Wheat</option>
            </select>
          </div>
          
          <div className="flex items-center">
            <label className="mr-2 text-gray-700">Forecast Days:</label>
            <select 
              className="border rounded p-2 bg-white"
              value={forecastDays}
              onChange={(e) => setForecastDays(parseInt(e.target.value))}
            >
              <option value="3">3 days</option>
              <option value="7">7 days</option>
              <option value="14">14 days</option>
              <option value="30">30 days</option>
            </select>
          </div>
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
        <div className="bg-white rounded-lg shadow p-4">
          <h3 className="text-lg font-semibold text-gray-700">MSE</h3>
          <p className="text-3xl font-bold text-blue-600">{metrics.mse}</p>
          <p className="text-sm text-gray-500">Mean Squared Error</p>
        </div>
        
        <div className="bg-white rounded-lg shadow p-4">
          <h3 className="text-lg font-semibold text-gray-700">RMSE</h3>
          <p className="text-3xl font-bold text-blue-600">{metrics.rmse}</p>
          <p className="text-sm text-gray-500">Root Mean Squared Error</p>
        </div>
        
        <div className="bg-white rounded-lg shadow p-4">
          <h3 className="text-lg font-semibold text-gray-700">MAE</h3>
          <p className="text-3xl font-bold text-blue-600">{metrics.mae}</p>
          <p className="text-sm text-gray-500">Mean Absolute Error</p>
        </div>
      </div>
      
      <div className="bg-white rounded-lg shadow p-6 mb-6">
        <h2 className="text-xl font-semibold text-gray-800 mb-4">Price Prediction Chart</h2>
        <div className="h-64 md:h-80">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={predictionData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="actualPrice" stroke="#2563eb" name="Actual Price" strokeWidth={2} />
              <Line type="monotone" dataKey="predictedPrice" stroke="#dc2626" name="Predicted Price" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-semibold text-gray-800 mb-4">Weather Feature Impact</h2>
          <div className="flex items-center mb-4">
            <label className="mr-2 text-gray-700">Select Feature:</label>
            <select 
              className="border rounded p-2 bg-white"
              value={selectedFeature}
              onChange={(e) => setSelectedFeature(e.target.value)}
            >
              <option value="t2m">Temperature (T2M)</option>
              <option value="cloudAmt">Cloud Amount</option>
              <option value="windSpeed">Wind Speed</option>
            </select>
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={predictionData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis yAxisId="left" orientation="left" stroke="#2563eb" />
                <YAxis yAxisId="right" orientation="right" stroke="#10b981" />
                <Tooltip />
                <Legend />
                <Line yAxisId="left" type="monotone" dataKey="predictedPrice" stroke="#2563eb" name="Price" />
                <Line yAxisId="right" type="monotone" dataKey={selectedFeature} stroke="#10b981" name={selectedFeature} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
        
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-semibold text-gray-800 mb-4">Forecast Accuracy</h2>
          <table className="min-w-full">
            <thead>
              <tr>
                <th className="py-2 px-4 border-b text-left">Date</th>
                <th className="py-2 px-4 border-b text-right">Actual</th>
                <th className="py-2 px-4 border-b text-right">Predicted</th>
                <th className="py-2 px-4 border-b text-right">Diff (%)</th>
              </tr>
            </thead>
            <tbody>
              {predictionData.slice(0, 5).map((data, index) => {
                const diffPercent = ((data.predictedPrice - data.actualPrice) / data.actualPrice * 100).toFixed(2);
                return (
                  <tr key={index}>
                    <td className="py-2 px-4 border-b">{data.date}</td>
                    <td className="py-2 px-4 border-b text-right">${data.actualPrice.toFixed(2)}</td>
                    <td className="py-2 px-4 border-b text-right">${data.predictedPrice.toFixed(2)}</td>
                    <td className={`py-2 px-4 border-b text-right ${parseFloat(diffPercent) > 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {diffPercent}%
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default PricePredictionDashboard;