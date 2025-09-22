import { useState, useEffect } from "react";
import {Container, Row, Col, Form, Button, Alert, Table, Card, Spinner, ButtonGroup} from "react-bootstrap";
import axios from "axios";
import {useNavigate } from "react-router-dom"; // For navigation
const Datasets = () => {
    const [activeTab, setActiveTab] = useState<'upload' | 'huggingface'>('upload');

    const [file, setFile] = useState<File | null>(null);
    const [datasets, setDatasets] = useState<any[]>([]);
    const [error, setError] = useState<string | null>(null);
    const [success, setSuccess] = useState<string | null>(null);
    const navigate = useNavigate(); // Define useNavigate correctly
    const [hfDatasetName, setHfDatasetName] = useState("");
    const [isImporting, setIsImporting] = useState(false);
    const handleDeleteDataset = async (datasetId: string) => {
        try {
            await axios.delete(
                `http://localhost:5000/api/delete_dataset/${datasetId}`,
                { withCredentials: true }
            );
            fetchDatasets(); // Refresh the list
            setSuccess("Dataset deleted successfully");
            setError(null);
        } catch (err) {
            setError("Failed to delete dataset");
            setSuccess(null);
        }
    };
    const handleHfImport = async () => {
        if (!hfDatasetName) {
            setError("Please enter a Hugging Face dataset name");
            return;
        }

        setIsImporting(true);
        try {
            const response = await axios.post(
                "http://localhost:5000/api/import_hf_dataset",
                { dataset_name: hfDatasetName },
                { withCredentials: true }
            );

            setSuccess(response.data.message);
            setError(null);
            fetchDatasets(); // Refresh list
            setHfDatasetName("");
        } catch (err) {
            setError("Failed to import dataset");
        } finally {
            setIsImporting(false);
        }
    };
    // Handle file selection
    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            setFile(e.target.files[0]);
        }
    };
    const handleViewDataset = (datasetId: string) => {
        navigate(`/dataset/${datasetId}`);
    };

    // Upload dataset to backend
    const handleUpload = async () => {
        if (!file) {
            setError("Please select a file.");
            return;
        }

        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await axios.post("http://localhost:5000/api/upload_dataset", formData, {
                headers: { "Content-Type": "multipart/form-data" },
                withCredentials: true,
            });

            setSuccess(response.data.message);
            setError(null);
            fetchDatasets(); // Refresh dataset list after upload
        } catch (err) {
            setError("Failed to upload dataset.");
            setSuccess(null);
        }
    };

    // Fetch uploaded datasets from backend
    const fetchDatasets = async () => {
        try {
            const response = await axios.get("http://localhost:5000/api/datasets", {
                withCredentials: true,
            });
            setDatasets(response.data.datasets);
            console.log(response.data.datasets);
        } catch (err) {
            setError("Failed to load datasets.");
        }
    };


    useEffect(() => {
        fetchDatasets();
    }, []);

    // Format uploaded_at date properly
    const formatDate = (dateString: string) => {
        if (!dateString) return "Unknown"; // Handle missing dates
        const date = new Date(dateString);
        return isNaN(date.getTime()) ? "Invalid Date" : date.toLocaleString();
    };

    return (
        <Container className="py-5">
            <Row className="justify-content-center mb-4">
                <Col md={8} className="text-center">
                    <h2 className="mb-4">Dataset Management</h2>
                    <div className="d-flex justify-content-center mb-4">
                        <ButtonGroup>
                            <Button
                                variant={activeTab === 'upload' ? 'dark' : 'outline-dark'}
                                onClick={() => setActiveTab('upload')}
                                className="mx-2"
                            >
                                <i className="bi bi-upload me-2"></i>
                                Upload CSV
                            </Button>
                            <Button
                                variant={activeTab === 'huggingface' ? 'dark' : 'outline-dark'}
                                onClick={() => setActiveTab('huggingface')}
                                className="mx-2"
                            >
                                <i className="bi bi-cloud-download me-2"></i>
                                Hugging Face
                            </Button>
                        </ButtonGroup>
                    </div>

                    {error && <Alert variant="danger">{error}</Alert>}
                    {success && <Alert variant="success">{success}</Alert>}

                    {/* Upload CSV Section */}
                    {activeTab === 'upload' && (
                        <Card className="mb-4 shadow border-dark ">
                            <Card.Body>
                                <Form.Group controlId="formFile" className="mb-4">
                                    <Form.Label>Select CSV File</Form.Label>
                                    <Form.Control
                                        type="file"
                                        accept=".csv"
                                        onChange={handleFileChange}
                                        className="border-secondary  "
                                    />
                                </Form.Group>
                                <Button
                                    variant="dark"
                                    className="w-100"
                                    onClick={handleUpload}
                                    disabled={!file}
                                >
                                    <i className="bi bi-cloud-arrow-up me-2"></i>
                                    Upload Dataset
                                </Button>
                            </Card.Body>
                        </Card>
                    )}

                    {/* Hugging Face Section */}
                    {activeTab === 'huggingface' && (
                        <Card className="mb-4 border-dark shadow">
                            <Card.Body>
                                <Form.Group controlId="formHFDataset" className="mb-3">
                                    <Form.Label>Hugging Face Dataset</Form.Label>
                                    <Form.Control
                                        type="text"
                                        value={hfDatasetName}
                                        onChange={(e) => setHfDatasetName(e.target.value)}
                                        placeholder="e.g. 'glue', 'sst2', 'imdb'"
                                        className="border-secondary "
                                    />
                                    <Form.Text className="text-muted">
                                        Enter dataset identifier from Hugging Face Hub
                                    </Form.Text>
                                </Form.Group>
                                <Button
                                    variant="dark"
                                    className="w-100"
                                    onClick={handleHfImport}
                                    disabled={isImporting || !hfDatasetName}
                                >
                                    {isImporting ? (
                                        <Spinner animation="border" size="sm" className="me-2" />
                                    ) : (
                                        <i className="bi bi-download me-2"></i>
                                    )}
                                    {isImporting ? "Importing..." : "Import Dataset"}
                                </Button>
                            </Card.Body>
                        </Card>
                    )}
                </Col>
            </Row>

            {/* Dataset List */}
            <Row className="justify-content-center">
                <Col md={10}>
                    <Card className="border shadow">
                        <Card.Body>
                            <Card.Title className="text-center mb-4">
                                <i className="bi bi-database me-2"></i>
                                Managed Datasets
                            </Card.Title>
                            <Table striped bordered hover >
                                <thead>
                                <tr>
                                    <th>#</th>
                                    <th>Filename</th>
                                    <th>Source</th>
                                    <th>Actions</th>
                                </tr>
                                </thead>
                                <tbody>
                                {datasets.map((dataset, index) => (
                                    <tr key={dataset._id}>
                                        <td>{index + 1}</td>
                                        <td>
                                            <Button
                                                variant="link"
                                                className="text-info p-0"
                                                onClick={() => handleViewDataset(dataset._id)}
                                            >
                                                <i className="bi bi-file-earmark-spreadsheet me-2"></i>
                                                {dataset.filename}
                                            </Button>
                                        </td>
                                        <td>{dataset.source}</td>
                                        <td>
                                            <Button
                                                variant="outline-danger"
                                                size="sm"
                                                onClick={() => handleDeleteDataset(dataset._id)}
                                            >
                                                <i className="bi bi-trash"></i>
                                            </Button>
                                        </td>
                                    </tr>
                                ))}
                                </tbody>
                            </Table>
                        </Card.Body>
                    </Card>
                </Col>
            </Row>
        </Container>
    );
};

export default Datasets;