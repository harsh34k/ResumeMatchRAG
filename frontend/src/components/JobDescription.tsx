// src/components/JobDescription.tsx
import React from 'react';

interface JobDescriptionProps {
    setJobDescription: React.Dispatch<React.SetStateAction<string>>;
}

const JobDescription: React.FC<JobDescriptionProps> = ({ setJobDescription }) => {
    return (
        <div className="space-y-4">
            <textarea
                onChange={(e) => setJobDescription(e.target.value)}
                placeholder="Enter job description here..."
                rows={6}
                className="w-full p-3 border border-gray-300 rounded-md"
            />
        </div>
    );
};

export default JobDescription;
