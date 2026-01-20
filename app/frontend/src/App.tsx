// import './App.css'
//
// function App() {
//     return (
//    <main className="py-8 bg-red-50 h-screen">
//        <h1>TODDO</h1>
//    </main>
//   )
// }
//
// export default App

import { useState } from 'react';
import { Layout } from './components/Layout';
import { FileUploader } from './components/FileUploader';
import { ImageGrid } from './components/ImageGrid';
import { ActionButton } from './components/ActionButton';
import { SummaryPanel } from './components/SummaryPanel';

function App() {
    const [images, setImages] = useState<string[]>([]);
    const [isProcessing, setIsProcessing] = useState(false);
    const [hasResults, setHasResults] = useState(false);
    const [volume, setVolume] = useState(0);

    const handleFilesSelected = (fileList: FileList | null) => {
        if (fileList) {
            const newImages = Array.from(fileList).map((file) => URL.createObjectURL(file));
            setImages((prev) => [...prev, ...newImages]);
            // Reset results when new images are added
            setHasResults(false);
        }
    };

    const handleRemoveImage = (index: number) => {
        setImages((prev) => prev.filter((_, i) => i !== index));
        if (images.length <= 1) setHasResults(false);
    };

    const handleMarkTumors = () => {
        if (images.length === 0) return;

        setIsProcessing(true);

        // Simulate complex calculation delay
        setTimeout(() => {
            setIsProcessing(false);
            setHasResults(true);
            // Simulate a calculated volume based on random logic for demo
            setVolume(Math.random() * 10 + 2.5);
        }, 2000);
    };

    return (
         // <div className="bg-red-500 w-full h-screen" />
        <Layout>
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 h-full">
                {/* Left Column: Input and Visualization */}
                <div className="lg:col-span-8 flex flex-col gap-8">
                    <section className="space-y-4">
                        <div className="flex items-center justify-between">
                            <h2 className="text-xl font-semibold text-medical-900 tracking-tight">Image Acquisition</h2>
                            <span className="text-sm text-medical-500 font-medium px-3 py-1 bg-white rounded-full border border-medical-200">
                {images.length} Scan{images.length !== 1 && 's'} Loaded
              </span>
                        </div>

                        <FileUploader onFilesSelected={handleFilesSelected} />
                    </section>

                    <section className="space-y-4 min-h-[200px]">
                        {images.length > 0 && (
                            <>
                                <h3 className="text-lg font-medium text-medical-800">Scan Sequence</h3>
                                <ImageGrid images={images} onRemove={handleRemoveImage} />
                            </>
                        )}
                    </section>
                </div>

                {/* Right Column: Controls and Results */}
                <div className="lg:col-span-4 flex flex-col gap-6">
                    {/*<div className="bg-white p-6 rounded-2xl border border-medical-200 shadow-sm sticky top-24">*/}
                    <div className="bg-blue-700 p-6 rounded-2xl border border-medical-200 shadow-sm sticky top-24">
                        <div className="mb-6">
                            <h3 className="text-lg font-semibold text-medical-900 mb-2">Actions</h3>
                            <p className="text-sm text-medical-500 leading-relaxed">
                                Initiate the segmentation algorithm to identify fluid regions and calculate volumetric data.
                            </p>
                        </div>

                        <ActionButton
                            onClick={handleMarkTumors}
                            isLoading={isProcessing}
                            disabled={images.length === 0}
                            label="Mark Tumors / Oznacz Guzy"
                        />

                        <div className="my-8 border-t border-medical-100" />

                        <div className="h-[400px]">
                            <SummaryPanel hasResults={hasResults} volume={volume} />
                        </div>
                    </div>
                </div>
            </div>
        </Layout>
    );
}

export default App;