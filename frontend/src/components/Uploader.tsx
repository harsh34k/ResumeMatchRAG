// src/components/Uploader.tsx
import React, { useState } from 'react';
import axios from 'axios';

interface UploaderProps {
    setSessionId: React.Dispatch<React.SetStateAction<string | null>>;
}

const Uploader: React.FC<UploaderProps> = ({ setSessionId }) => {
    const [files, setFiles] = useState<File[]>([]);
    const [uploading, setUploading] = useState(false);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files) {
            setFiles(Array.from(e.target.files));
        }
    };

    const handleUpload = async () => {
        if (files.length === 0) return alert('Please select files to upload.');

        setUploading(true);
        const formData = new FormData();
        files.forEach(file => formData.append('files', file));

        try {
            const response = await axios.post('http://localhost:8000/upload_pdfs/', formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
            });
            setSessionId(response.data.session_id);
            alert('Files uploaded successfully!');
        } catch (error) {
            console.error(error);
            alert('Error uploading files.');
        } finally {
            setUploading(false);
        }
    };

    return (
        <div className="space-y-4">
            <input
                type="file"
                multiple
                onChange={handleFileChange}
                className="p-2 border border-gray-300 rounded-md"
            />
            <button
                onClick={handleUpload}
                disabled={uploading}
                className="w-full py-3 bg-gradient-to-r from-primary to-blue-600 text-white font-semibold rounded-lg shadow-md hover:from-blue-600 hover:to-primary transition-all duration-300"
            >
                {uploading ? 'Uploading...' : 'Upload Resumes'}
            </button>
        </div>
    );
};

export default Uploader;
