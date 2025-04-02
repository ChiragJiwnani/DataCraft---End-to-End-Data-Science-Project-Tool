// src/App.js
import React, { useState } from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Dashboard from './components/dashboard';
import DataUpload from './components/dataupload';
import DataCleaning from './components/datacleaning';
import ModelSelection from './components/modelselection';
import ModelEvaluation from './components/modelevaluation';
import './App.css';

function App() {
  const [uploadedFile, setUploadedFile] = useState(null); // State to keep track of the uploaded file

  return (
    <Router>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/upload-data" element={<DataUpload setUploadedFile={setUploadedFile} />} />
        <Route path="/data-cleaning" element={<DataCleaning uploadedFile={uploadedFile} />} />
        <Route path="/train-model" element={<ModelSelection />} />
        <Route path="/evaluate" element={<ModelEvaluation/>} />
      </Routes>
    </Router>
  );
}

export default App;
