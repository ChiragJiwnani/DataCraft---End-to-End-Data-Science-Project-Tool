import React, { useState } from 'react';

const UploadResults = () => {
    const [file, setFile] = useState(null);
    const [modelName, setModelName] = useState('');
    const [message, setMessage] = useState('');

    const handleFileChange = (e) => {
        setFile(e.target.files[0]);
    };

    const handleModelChange = (e) => {
        setModelName(e.target.value);
    };

    const handleUpload = async () => {
        if (!file || !modelName) {
            setMessage('Please select a model and a file to upload.');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);
        formData.append('model_name', modelName);

        try {
            const response = await fetch('http://localhost:5000/upload-results', {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();
            setMessage(data.message);
        } catch (error) {
            setMessage('Error uploading results.');
        }
    };

    return (
        <div>
            <h2>Upload Model Evaluation Results</h2>
            <input type="text" placeholder="Model Name" onChange={handleModelChange} />
            <input type="file" onChange={handleFileChange} />
            <button onClick={handleUpload}>Upload</button>
            {message && <p>{message}</p>}
        </div>
    );
};

export default UploadResults;
