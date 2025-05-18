import { useEffect, useState } from "react";
import { useParams, Link, useNavigate } from "react-router-dom";
import { Container, Row, Col, Table, Button, Alert, Spinner, Card, Badge, Pagination, Form, Modal } from "react-bootstrap";
import axios from "axios";

import { useProvider } from "../modules/provider";

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
    const { provider, model } = useProvider();

    // Pagination states
    const [currentPage, setCurrentPage] = useState(1);
    const [itemsPerPage, setItemsPerPage] = useState(20);
    const [selectedEntry, setSelectedEntry] = useState<any | null>(null);
    const [showModal, setShowModal] = useState(false);

    const formatDate = (dateString: string) => {
        return new Date(dateString).toLocaleString();
    };

    // Calculate pagination values
    const indexOfLastItem = currentPage * itemsPerPage;
    const indexOfFirstItem = indexOfLastItem - itemsPerPage;
    const currentItems = dataset?.data ? dataset.data.slice(indexOfFirstItem, indexOfLastItem) : [];
    const totalPages = dataset?.data ? Math.ceil(dataset.data.length / itemsPerPage) : 0;

    // Generate pagination items
    const paginationItems = [];
    for (let number = 1; number <= totalPages; number++) {
        paginationItems.push(
            <Pagination.Item 
                key={number} 
                active={number === currentPage}
                onClick={() => setCurrentPage(number)}
            >
                {number}
            </Pagination.Item>
        );
    }

    // Handle items per page change
    const handleItemsPerPageChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
        setItemsPerPage(Number(e.target.value));
        setCurrentPage(1); // Reset to first page when changing items per page
    };

    // Handle entry click
    const handleEntryClick = (entry: any) => {
        setSelectedEntry(entry);
        setShowModal(true);
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
                { method:method,provider: provider, model: model },
                { withCredentials: true }
            );

            // Navigate to classification dashboard after successful classification
            navigate(`/datasets/${datasetId}/classifications/${response.data.classification_id}`);

        } catch (err) {
            console.log(err);
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
        <Container fluid>
            <Row className="py-4">
                {/* Classification Sidebar */}
                <Col md={2} className="border-end border-2">
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
                <Col md={10}>
                    {loading ? (
                        <div className="text-center">
                            <Spinner animation="border" />
                        </div>
                    ) : error ? (
                        <Alert variant="danger">{error}</Alert>
                    ) : (
                        <>
                            <div className="d-flex justify-content-between align-items-center mb-4">
                                <h2 className="mb-0">{dataset?.filename}</h2>
                                <div className="d-flex align-items-center">
                                    <span className="me-3">Items per page:</span>
                                    <Form.Select 
                                        size="sm" 
                                        style={{ width: '100px' }}
                                        value={itemsPerPage}
                                        onChange={handleItemsPerPageChange}
                                        className="me-2"
                                    >
                                        <option value={5}>5</option>
                                        <option value={10}>10</option>
                                        <option value={20}>20</option>
                                        <option value={50}>50</option>
                                    </Form.Select>
                                </div>
                            </div>

                            {dataset?.data && dataset.data.length > 0 ? (
                                <>
                                    <Card className="shadow-sm mb-4">
                                        <Table hover responsive className="mb-0">
                                            <thead className="bg-light">
                                                <tr>
                                                    {Object.keys(dataset.data[0]).map((key) => (
                                                        <th key={key} className="px-4 py-3">{key}</th>
                                                    ))}
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {currentItems.map((row, index) => (
                                                    <tr 
                                                        key={index}
                                                        onClick={() => handleEntryClick(row)}
                                                        style={{ cursor: 'pointer' }}
                                                        className="hover-highlight"
                                                    >
                                                        {Object.values(row).map((value, i) => (
                                                            <td key={i} className="px-4 py-3 text-truncate" style={{ maxWidth: '300px' }}>
                                                                {String(value)}
                                                            </td>
                                                        ))}
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </Table>
                                    </Card>

                                    <div className="d-flex justify-content-between align-items-center">
                                        <div className="text-muted">
                                            Showing {indexOfFirstItem + 1} to {Math.min(indexOfLastItem, dataset.data.length)} of {dataset.data.length} entries
                                        </div>
                                        <Pagination className="mb-0">
                                            <Pagination.First onClick={() => setCurrentPage(1)} disabled={currentPage === 1} />
                                            <Pagination.Prev onClick={() => setCurrentPage(curr => Math.max(curr - 1, 1))} disabled={currentPage === 1} />
                                            {paginationItems}
                                            <Pagination.Next onClick={() => setCurrentPage(curr => Math.min(curr + 1, totalPages))} disabled={currentPage === totalPages} />
                                            <Pagination.Last onClick={() => setCurrentPage(totalPages)} disabled={currentPage === totalPages} />
                                        </Pagination>
                                    </div>
                                </>
                            ) : (
                                <p className="text-center">No data available.</p>
                            )}
                        </>
                    )}
                </Col>
            </Row>

            {/* Modal for viewing full entry details */}
            <Modal show={showModal} onHide={() => setShowModal(false)} size="lg">
                <Modal.Header closeButton>
                    <Modal.Title>Entry Details</Modal.Title>
                </Modal.Header>
                <Modal.Body>
                    {selectedEntry && (
                        <div className="p-4">
                            {Object.entries(selectedEntry).map(([key, value]) => (
                                <div key={key} className="mb-4">
                                    <h6 className="text-muted mb-2">{key}</h6>
                                    <div style={{ whiteSpace: 'pre-wrap' }} className="p-3 bg-light rounded">
                                        {String(value)}
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </Modal.Body>
                <Modal.Footer>
                    <Button variant="secondary" onClick={() => setShowModal(false)}>
                        Close
                    </Button>
                </Modal.Footer>
            </Modal>
        </Container>
    );
};

// Add some custom styles
const styles = `
.hover-highlight:hover {
    background-color: rgba(0,0,0,0.05);
}
`;

// Add styles to document
const styleSheet = document.createElement("style");
styleSheet.innerText = styles;
document.head.appendChild(styleSheet);

export default DatasetView;