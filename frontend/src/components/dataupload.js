// dataUpload.js
import React, { useState } from 'react';
import DataCleaning from './datacleaning';

function DataUpload() {
  const [uploadedFilePath, setUploadedFilePath] = useState('');
  const [uploadError, setUploadError] = useState(''); // Track upload errors
  const [loading, setLoading] = useState(false);
  const [augmentationMessage, setAugmentationMessage] = useState('');
  const [augmenting, setAugmenting] = useState(false);

  // Handle file upload
  const handleUpload = async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    setLoading(true);
    setAugmentationMessage(''); // Reset augmentation message

    try {
      const response = await fetch('http://localhost:5000/upload-data', {
        method: 'POST',
        body: formData,
      });

      // Check if the response is OK and content type is JSON
      if (response.ok && response.headers.get("content-type")?.includes("application/json")) {
        const result = await response.json();
        setUploadedFilePath(result.filepath);
        console.log('Uploaded file path:', result.filepath); // Debugging
        setUploadError(''); // Clear any previous errors
      } else {
        // Parse error message if JSON, else fallback to text
        const errorText = await response.text();
        console.error('Upload failed:', errorText);
        setUploadError('Upload failed: ' + errorText);
      }
    } catch (error) {
      console.error('Error uploading the file:', error);
      setUploadError('Error uploading the file: ' + error.message);
    }
    finally {
      setLoading(false);
    }
  };

   // Handle data augmentation request
  const handleAugmentation = async () => {
    if (!uploadedFilePath) {
      setAugmentationMessage('Please upload a file first.');
      return;
    }
    
    setAugmenting(true);
    setAugmentationMessage('');
    
    try {
      const response = await fetch('http://localhost:5000/augment-data', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filepath: uploadedFilePath }),
      });

      const result = await response.json();

      if (response.ok) {
        setAugmentationMessage('Data augmentation completed successfully!');
      } else {
        setAugmentationMessage('Augmentation failed: ' + result.error);
      }
    } catch (error) {
      console.error('Error during augmentation:', error);
      setAugmentationMessage('Augmentation failed: ' + error.message);
    } finally {
      setAugmenting(false);
    }
  };


  return (
    <div>
      {/* <h2 className='h2heading'>Data Upload</h2> */}
      <h1>Data Upload</h1>
      <div className='inputbox'>

      <input type="file" onChange={(e) => handleUpload(e.target.files[0])} accept=".csv"  disabled={loading}/>
          </div>

          <div className='datacleaningblock'>
      {loading && <p>Uploading file...</p>}
      {uploadedFilePath && !loading &&(
        <div>
          <h3>File Uploaded: {uploadedFilePath}</h3>
          {/* Pass the uploadedFilePath to the DataCleaning component */}
          <DataCleaning uploadedFilePath={uploadedFilePath} />
        </div>
      )}
      </div>
      
      {/* Augmentation Button */}
      {uploadedFilePath && (
        <button onClick={handleAugmentation} disabled={augmenting}>
          {augmenting ? 'Augmenting...' : 'Augment Data with GAN'}
        </button>
      )}
      
      {/* Display augmentation message */}
      {augmentationMessage && <p style={{ color: 'green' }}>{augmentationMessage}</p>}

      {/* Display upload error message */}
      {uploadError && <p style={{ color: 'red' }}>{uploadError}</p>}

      <div className="footer">
        <div>
          <a href="/train-model">
            <button>
              Model Selection
              <svg class="w-6 h-6 text-gray-800 dark:text-white" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="none" viewBox="0 0 24 24">
  <path stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="m9 5 7 7-7 7"/>
</svg>

            </button>
          </a>
        </div>
        <div>
          <a href="/evaluate">
            <button>
              Model Evaluation
              <svg class="w-6 h-6 text-gray-800 dark:text-white" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="none" viewBox="0 0 24 24">
  <path stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="m7 16 4-4-4-4m6 8 4-4-4-4"/>
</svg>


            </button>
          </a>
        </div>  
      </div>
      
    </div>
    
  );
  
}

export default DataUpload;
