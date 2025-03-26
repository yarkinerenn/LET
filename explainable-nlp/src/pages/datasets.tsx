import { useState, useEffect } from "react";
import { Container, Row, Col, Form, Button, Alert, Table } from "react-bootstrap";
import axios from "axios";
import {useNavigate } from "react-router-dom"; // For navigation

const Datasets = () => {
    const [file, setFile] = useState<File | null>(null);
    const [datasets, setDatasets] = useState<any[]>([]);
    const [error, setError] = useState<string | null>(null);
    const [success, setSuccess] = useState<string | null>(null);
    const navigate = useNavigate(); // Define useNavigate correctly

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
        } catch (err) {
            setError("Failed to load datasets.");
        }
    };

    // Delete dataset from backend
    const handleDelete = async (datasetId: string) => {
        try {
            await axios.delete(`http://localhost:5000/api/delete_dataset/${datasetId}`, {
                withCredentials: true,
            });

            setSuccess("Dataset deleted successfully.");
            setError(null);
            fetchDatasets(); // Refresh dataset list after deletion
        } catch (err) {
            setError("Failed to delete dataset.");
            setSuccess(null);
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
            <Row className="justify-content-center">
                <Col md={8}>
                    <h2 className="text-center mb-4">Upload Dataset</h2>

                    {error && <Alert variant="danger">{error}</Alert>}
                    {success && <Alert variant="success">{success}</Alert>}

                    <Form.Group controlId="formFile" className="mb-3">
                        <Form.Label>Select a CSV File</Form.Label>
                        <Form.Control type="file" accept=".csv" onChange={handleFileChange} />
                    </Form.Group>

                    <Button variant="primary" className="w-100 mb-4" onClick={handleUpload}>
                        Upload Dataset
                    </Button>

                    <h3 className="text-center mb-3">Uploaded Datasets</h3>
                    <Table striped bordered hover>
                        <thead>
                        <tr>
                            <th>#</th>
                            <th>Filename</th>
                            <th>Uploaded At</th>
                            <th>Actions</th>
                        </tr>
                        </thead>
                        <tbody>
                        {datasets.map((dataset, index) => (
                            <tr key={dataset._id}>
                                <td>{index + 1}</td>
                                <td>
                                    <Button variant="link" onClick={() => handleViewDataset(dataset._id)}>
                                        {dataset.filename}
                                    </Button>
                                </td>
                                <td>{new Date(dataset.uploaded_at).toLocaleString()}</td>
                                <td>
                                    <Button
                                        variant="danger"
                                        size="sm"
                                        onClick={async () => {
                                            await axios.delete(`http://localhost:5000/api/delete_dataset/${dataset._id}`, {
                                                withCredentials: true,
                                            });
                                            fetchDatasets();
                                        }}
                                    >
                                        Delete
                                    </Button>
                                </td>
                            </tr>
                        ))}
                        </tbody>
                    </Table>
                </Col>
            </Row>
        </Container>
    );
};

export default Datasets;