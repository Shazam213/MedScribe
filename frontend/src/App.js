import React, { useState } from 'react';
import axios from 'axios';
import { Upload, Sidebar, List, X, User, Bell, Settings, LogOut, Moon, Sun } from 'lucide-react';

const App = () => {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [uploadedDocuments, setUploadedDocuments] = useState([]);
  const [darkMode, setDarkMode] = useState(false);
  const [ocrText, setOcrText] = useState(''); // State to store OCR result
  const [error, setError] = useState(null); // State to handle errors

  // Handle file upload to backend
  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (file) {
      const formData = new FormData();
      formData.append('file', file); // Add the selected file to FormData
      
      try {
        const response = await axios.post('http://localhost:5000/upload', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        });
        // Set the OCR response text from the backend
        setOcrText(response.data.ocr_text);
        setUploadedDocuments([...uploadedDocuments, file.name]);
        setIsModalOpen(false); // Close the modal after upload
      } catch (err) {
        setError('Failed to upload the document or process OCR. Please try again.');
        console.error(err);
      }
    }
  };

  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
  };

  const handleRemove = (indexToRemove) => {
    setUploadedDocuments(uploadedDocuments.filter((_, index) => index !== indexToRemove));
  };

  // Handle changes in the textarea to update OCR text state
  const handleTextChange = (event) => {
    setOcrText(event.target.value);
  };
  const handleSave = () => {
    console.log('Save button clicked. OCR text:', ocrText);
    // You can add logic here to save the text or trigger an action
  };


  return (
    <div className={`flex h-screen ${darkMode ? 'dark' : ''}`}>
      {/* Sidebar */}
      <div className="w-64 bg-indigo-700 dark:bg-gray-800 text-white">
        <div className="p-6">
          <h1 className="text-2xl font-bold mb-6">MedScribe</h1>
          <div className="flex items-center mb-6">
            <div className="w-12 h-12 rounded-full bg-indigo-500 dark:bg-indigo-600 flex items-center justify-center text-xl font-bold mr-3">
              SM
            </div>
            <div>
              <h2 className="font-semibold">Soham Mulye</h2>
              <p className="text-indigo-300 dark:text-indigo-400 text-sm">Prime User</p>
            </div>
          </div>
        </div>
        <nav className="mt-4">
          {[
            { icon: Sidebar, label: 'Dashboard' },
            { icon: List, label: 'My Documents' },
            { icon: User, label: 'Profile' },
            { icon: Bell, label: 'Notifications' },
            { icon: Settings, label: 'Settings' },
          ].map((item, index) => (
            <button
              key={index}
              className="w-full text-left px-6 py-3 hover:bg-indigo-600 dark:hover:bg-gray-700 transition duration-150 flex items-center"
            >
              <item.icon className="mr-3 h-5 w-5" />
              {item.label}
            </button>
          ))}
        </nav>
        <div className="absolute bottom-0 w-64 p-6">
          <button className="w-full text-left px-6 py-3 hover:bg-indigo-600 dark:hover:bg-gray-700 transition duration-150 flex items-center text-indigo-300 dark:text-indigo-400">
            <LogOut className="mr-3 h-5 w-5" />
            Log Out
          </button>
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 overflow-hidden bg-gray-100 dark:bg-gray-900">
        <header className="bg-white dark:bg-gray-800 shadow-sm">
          <div className="max-w-7xl mx-auto py-4 px-4 sm:px-6 lg:px-8 flex justify-between items-center">
            <h1 className="text-2xl font-semibold text-gray-900 dark:text-white">Document Manager</h1>
            <button
              onClick={toggleDarkMode}
              className="p-2 rounded-full bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600 transition duration-150"
            >
              {darkMode ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
            </button>
          </div>
        </header>
        <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
          <div className="px-4 py-6 sm:px-0">
            <button
              className="bg-indigo-600 text-white px-4 py-2 rounded-md hover:bg-indigo-700 transition duration-150 flex items-center"
              onClick={() => setIsModalOpen(true)}
            >
              <Upload className="mr-2 h-5 w-5" />
              Upload Document
            </button>

            {/* List of uploaded documents */}
            <div className="mt-8">
              <h2 className="text-xl font-semibold mb-4 text-gray-800 dark:text-white">Uploaded Documents</h2>
              {uploadedDocuments.length > 0 ? (
                <ul className="space-y-3">
                  {uploadedDocuments.map((doc, index) => (
                    <li
                      key={index}
                      className="flex items-center justify-between bg-white dark:bg-gray-800 p-4 rounded-lg shadow-sm hover:shadow-md transition duration-150"
                    >
                      <span className="text-gray-700 dark:text-gray-300">{doc}</span>
                      <button
                        className="text-gray-400 hover:text-red-500 transition duration-150"
                        onClick={() => handleRemove(index)}
                      >
                        <X className="h-5 w-5" />
                      </button>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-gray-500 dark:text-gray-400 bg-white dark:bg-gray-800 p-4 rounded-lg shadow-sm">
                  No documents uploaded yet.
                </p>
              )}
            </div>

            {/* Editable OCR Result */}
            {ocrText && (
              <div className="mt-8">
                <h2 className="text-xl font-semibold mb-4 text-gray-800 dark:text-white">OCR Result</h2>
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow-md max-h-96 overflow-auto">  
                  <textarea
                    className="w-full h-64 p-3 border border-gray-300 dark:border-gray-700 dark:bg-gray-800 dark:text-white rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
                    value={ocrText}
                    onChange={handleTextChange}  // Capture user edits
                  />
                </div>
                {/* Save button */}
              <button
                onClick={handleSave}  // Trigger dummy save function
                className="mt-4 bg-indigo-600 text-white px-4 py-2 rounded-md hover:bg-indigo-700 transition duration-150 flex items-center"
              >
                Save
              </button>
              </div>

            )}

            {/* Display Errors if any */}
            {error && (
              <div className="mt-8 bg-red-100 dark:bg-red-800 p-4 rounded-lg shadow-sm text-red-600 dark:text-red-300">
                <p>{error}</p>
              </div>
            )}
          </div>
        </main>
      </div>

      {/* Upload Modal */}
      {isModalOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-xl max-w-md w-full transform transition-all duration-300 ease-in-out">
            <h2 className="text-2xl font-semibold mb-4 text-gray-800 dark:text-white">Upload a new document</h2>
            <input
              type="file"
              onChange={handleFileUpload}
              className="border border-gray-300 dark:border-gray-600 p-2 rounded-md w-full mb-4 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent dark:bg-gray-700 dark:text-white"
            />
            <div className="flex justify-end space-x-3">
              <button
                className="px-4 py-2 rounded-md text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 transition duration-150"
                onClick={() => setIsModalOpen(false)}
              >
                Cancel
              </button>
              <button
                className="bg-indigo-600 text-white px-4 py-2 rounded-md hover:bg-indigo-700 transition duration-150"
                onClick={() => setIsModalOpen(false)}
              >
                Upload
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;
