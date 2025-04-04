import { useEffect, useState } from "react";
import { useParams, Link,useNavigate } from "react-router-dom";
import { Container, Row, Col, Table, Button, Alert, Spinner, Card } from "react-bootstrap";
import axios from "axios";

const DatasetView = () => {
    const { datasetId } = useParams();
    const [dataset, setDataset] = useState<{ filename: string; data: any[] } | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [loading, setLoading] = useState(true);
    const [classifying, setClassifying] = useState<"llm" | "bert" | null>(null);
    const navigate = useNavigate();

    const handleClassification = async (method: "llm" | "bert") => {
        setClassifying(method);
        try {
            const response = await axios.post(
                `http://localhost:5000/api/classify/${datasetId}`,
                { method },
                { withCredentials: true }
            );

            // Navigate to classification dashboard after successful classification
            navigate(`/datasets/${datasetId}/classifications/${response.data.classification_id}`);

        } catch (err) {
            setError(`Classification using ${method.toUpperCase()} failed.`);
        } finally {
            setClassifying(null);
        }
    };

    useEffect(() => {
        const fetchDataset = async () => {
            try {
                const response = await axios.get(`http://localhost:5000/api/dataset/${datasetId}`, {
                    withCredentials: true,
                });
                setDataset(response.data);
            } catch (err) {
                setError("Failed to load dataset.");
            } finally {
                setLoading(false);
            }
        };

        fetchDataset();
    }, [datasetId]);

    return (
        <Container fluid className="py-5">
            <Row>
                {/* Classification Sidebar */}
                <Col md={3} className="border-end border-2 pe-4">
                    <Card className="border-0">
                        <Card.Body>
                            <Card.Title className="mb-4">Classification Methods</Card.Title>
                            <div className="d-grid gap-3">
                                <Button
                                    variant="primary"
                                    onClick={() => handleClassification("llm")}
                                    disabled={!dataset || !!classifying}
                                >
                                    {classifying === "llm" ? (
                                        <>
                                            <Spinner animation="border" size="sm" /> Classifying...
                                        </>
                                    ) : (
                                        "Classify with LLM"
                                    )}
                                </Button>
                                <Button
                                    variant="success"
                                    onClick={() => handleClassification("bert")}
                                    disabled={!dataset || !!classifying}
                                >
                                    {classifying === "bert" ? (
                                        <>
                                            <Spinner animation="border" size="sm" /> Classifying...
                                        </>
                                    ) : (
                                        "Classify with BERT"
                                    )}
                                </Button>
                            </div>
                        </Card.Body>
                    </Card>
                </Col>

                {/* Main Content */}
                <Col md={9}>
                    <Row className="mb-4">
                        <Col>
                            <Link to="/datasets">
                                <Button variant="secondary">← Back to Datasets</Button>
                            </Link>
                        </Col>
                    </Row>

                    {loading ? (
                        <div className="text-center">
                            <Spinner animation="border" />
                        </div>
                    ) : error ? (
                        <Alert variant="danger">{error}</Alert>
                    ) : (
                        <>
                            <h2 className="text-center mb-4">{dataset?.filename}</h2>
                            {dataset?.data && dataset.data.length > 0 ? (
                                <Table striped bordered hover responsive>
                                    <thead>
                                    <tr>
                                        {Object.keys(dataset.data[0]).map((key) => (
                                            <th key={key}>{key}</th>
                                        ))}
                                    </tr>
                                    </thead>
                                    <tbody>
                                    {dataset.data.map((row, index) => (
                                        <tr key={index}>
                                            {Object.values(row).map((value, i) => (
                                                <td key={i}>{String(value)}</td>
                                            ))}
                                        </tr>
                                    ))}
                                    </tbody>
                                </Table>
                            ) : (
                                <p className="text-center">No data available.</p>
                            )}
                        </>
                    )}
                </Col>
            </Row>
        </Container>
    );
};

export default DatasetView;