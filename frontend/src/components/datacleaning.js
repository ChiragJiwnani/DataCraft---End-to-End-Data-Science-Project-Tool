// datacleaning.js
import React, { useState, useEffect } from 'react';

function DataCleaning({ uploadedFilePath }) {
  const [cleaningLogs, setCleaningLogs] = useState([]);
  const [isCleaning, setIsCleaning] = useState(false);
  const [cleanedFilePath, setCleanedFilePath] = useState('');

  useEffect(() => {
    // Clear logs when file changes
    if (uploadedFilePath) {
      setCleaningLogs([]);
      setCleanedFilePath('');
      console.log('Received file path for cleaning:', uploadedFilePath); // Debugging
    }
  }, [uploadedFilePath]);

  // Function to log messages during cleaning
  const logMessage = (message) => {
    setCleaningLogs((prevLogs) => [...prevLogs, message]);
  };

  // Handle data cleaning
  const handleDataCleaning = async () => {
    if (!uploadedFilePath) {
      logMessage('No file available for cleaning. Please upload a file first.');
      return;
    }

    setIsCleaning(true);
    logMessage('Starting data cleaning...');

    try {
      const response = await fetch('http://localhost:5000/clean-data', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ filepath: uploadedFilePath }),
      });

      const result = await response.json();
      if (response.ok) {
        setCleanedFilePath(result.cleaned_filepath);
        logMessage('Data cleaned successfully.');
        logMessage(`Cleaned file saved at: ${result.cleaned_filepath}`);
      } else {
        logMessage(`Cleaning failed: ${result.error}`);
      }
    } catch (error) {
      logMessage('Error during data cleaning: ' + error.message);
    }

    setIsCleaning(false);
  };

  return (
    <div>
      {/* <h2 className='h2heading'>Data Cleaning</h2> */}
      <h1>Data Cleaning</h1>

      <button onClick={handleDataCleaning} disabled={isCleaning}>
        {isCleaning ? 'Cleaning Data...' : 'Clean Data'}
      </button>

      <div>
        <h4>Cleaning Logs:</h4>
        <div style={{ background: '#f5f5f5', padding: '10px', borderRadius: '5px', maxHeight: '200px', overflowY: 'scroll' }}>
          {cleaningLogs.length > 0 ? (
            cleaningLogs.map((log, index) => <p key={index}>{log}</p>)
          ) : (
            <p>No logs available yet.</p>
          )}
        </div>
      </div>

      {cleanedFilePath && (
        <div>
          <h4>Cleaned File Path:</h4>
          <p className='cleanedfilepath'>{cleanedFilePath}</p>
        </div>
      )}
    </div>
  );
}

export default DataCleaning;
