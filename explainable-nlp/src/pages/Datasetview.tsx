import { useEffect, useState } from "react";
import { useParams, Link,useNavigate } from "react-router-dom";
import { Container, Row, Col, Table, Button, Alert, Spinner, Card,Badge } from "react-bootstrap";
import axios from "axios";
interface ClassificationItem {
    _id: string;
    method: string;
    provider?: string;
    model?: string;
    created_at: string;
    stats: {
        total: number;
        positive: number;
        negative: number;
        accuracy?: number;
        precision?: number;
        recall?: number;
        f1_score?: number;
    };
}
const DatasetView = () => {
    const { datasetId } = useParams();
    const [dataset, setDataset] = useState<{ filename: string; data: any[] } | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [loading, setLoading] = useState(true);
    const [classifying, setClassifying] = useState<"llm" | "bert" | null>(null);
    const navigate = useNavigate();
    const [loadingClassifications, setLoadingClassifications] = useState(false);
    const [classifications, setClassifications] = useState<ClassificationItem[]>([]);
    const formatDate = (dateString: string) => {
        return new Date(dateString).toLocaleString();
    };
    const handleDeleteClassification = async (classificationId: string, e: React.MouseEvent) => {
        e.stopPropagation(); // Prevent navigation when clicking delete

        if (!window.confirm("Are you sure you want to delete this classification?")) {
            return;
        }

        try {
            await axios.delete(
                `http://localhost:5000/api/delete_classification/${classificationId}`,
                { withCredentials: true }
            );

            // Refresh the classifications list
            fetchClassifications();
        } catch (err) {
            setError("Failed to delete classification");
        }
    };

    // Helper for classification metrics display
    const getAccuracyBadge = (accuracy: number | undefined) => {
        if (accuracy === undefined) return null;

        let variant = "secondary";
        if (accuracy >= 0.8) variant = "success";
        else if (accuracy >= 0.6) variant = "warning";
        else variant = "danger";

        return <Badge bg={variant}>{(accuracy * 100).toFixed(1)}%</Badge>;
    };
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
    const fetchClassifications = async () => {
        setLoadingClassifications(true);
        try {
            const response = await axios.get(
                `http://localhost:5000/api/classifications/${datasetId}`,
                { withCredentials: true }
            );
            setClassifications(response.data.classifications);
        } catch (err) {
            console.error("Failed to load classifications:", err);
        } finally {
            setLoadingClassifications(false);
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
        fetchClassifications();

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
                    <Card className="border-0">
                        <Card.Body>
                            <Card.Title className="mb-3">Previous Classifications</Card.Title>
                            {loadingClassifications ? (
                                <div className="text-center py-3">
                                    <Spinner animation="border" size="sm" />
                                </div>
                            ) : classifications.length > 0 ? (
                                <div className="classifications-list">
                                    {classifications.map((classification) => (
                                        <Card
                                            key={classification._id}
                                            className="mb-2 classification-card"
                                            onClick={() => navigate(`/datasets/${datasetId}/classifications/${classification._id}`)}
                                            style={{ cursor: 'pointer' }}
                                        >
                                            <Card.Body className="p-3">
                                                <div className="d-flex justify-content-between align-items-start">
                                                    <div>
                                                        <div className="d-flex justify-content-between align-items-center mb-1">
                                                            <Badge bg={classification.method === "llm" ? "primary" : "success"}>
                                                                {classification.method.toUpperCase()}
                                                            </Badge>
                                                            {getAccuracyBadge(classification.stats.accuracy)}
                                                        </div>

                                                        {classification.method === "llm" && (
                                                            <div className="text-muted small mb-1">
                                                                {classification.provider} / {classification.model}
                                                            </div>
                                                        )}

                                                        <div className="small text-muted">
                                                            {formatDate(classification.created_at)}
                                                        </div>

                                                        <div className="small mt-1">
                                                            <span className="text-success me-2">Positive: {classification.stats.positive}</span>
                                                            <span className="text-danger">Negative: {classification.stats.negative}</span>
                                                        </div>
                                                    </div>
                                                    <Button
                                                        variant="outline-danger"
                                                        size="sm"
                                                        onClick={(e) => handleDeleteClassification(classification._id, e)}
                                                        className="ms-2"
                                                    >
                                                        Delete
                                                    </Button>
                                                </div>
                                            </Card.Body>
                                        </Card>
                                    ))}
                                </div>
                            ) : (
                                <p className="text-muted text-center">No previous classifications</p>
                            )}
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