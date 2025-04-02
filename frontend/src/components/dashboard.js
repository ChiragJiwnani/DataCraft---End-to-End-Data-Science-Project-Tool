// src/components/Dashboard.js
import React from 'react';
import { Link } from 'react-router-dom';
import dashboard from "./componentcss/dashboard.css"

function Dashboard() {
  return (
    <div className="dashboard">
      <h1>DataCraft Dashboard</h1>
      <div className="card-container">
        <Link to="/upload-data" className="card">
          <h2>Data  Cleaning & Analysis</h2>
          <p>Upload your datasets to start the data cleaning and analysis process.</p>
        </Link>
        {/* <Link to="/data-cleaning" className="card">
          <h2>Data Cleaning & Analysis</h2>
          <p>Access tools for data cleaning and analysis.</p>
        </Link> */}
        <Link to="/train-model" className="card">
          <h2>Model Selection</h2>
          <p>Choose and configure models for your data analysis.</p>
        </Link>
        <Link to="/evaluate" className="card">
          <h2>Evaluation</h2>
          <p>Evaluate the performance of your models.</p>
        </Link>
      </div>
    </div>
  );
}

export default Dashboard;
