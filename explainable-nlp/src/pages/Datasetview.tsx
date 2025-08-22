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
    const [classificationLimit, setClassificationLimit] = useState<number | null>(null);
    const { provider, model } = useProvider();
    const [dataType, setDataType] = useState<'sentiment' | 'legal'|'medical'|'ecqa'|'snarks'|'hotel'>('sentiment');
    const [exploreLoading, setExploreLoading] = useState(false);
    const handleExploreDataset = async () => {
        setExploreLoading(true);
        setError(null);
        try {
            const response = await axios.post(
                `http://localhost:5000/api/classification/empty/${datasetId}`,
                {},
                { withCredentials: true }
            );
            if (["medical"].includes((response.data.data_type || dataType).toLowerCase())) {
                navigate(`/datasets/${datasetId}/classifications_pub/${response.data.classification_id}/results/0`);
            } else if (["sentiment", "legal"].includes((response.data.data_type || dataType).toLowerCase())) {
                navigate(`/datasets/${datasetId}/classifications/${response.data.classification_id}/results/0`);
            } else {
                navigate(`/datasets/${datasetId}/classifications/${response.data.classification_id}`);
            }
        } catch (err) {
            setError("Failed to start exploration mode.");
        } finally {
            setExploreLoading(false);
        }
    };
    // Pagination states
    const [currentPage, setCurrentPage] = useState(1);
    const [itemsPerPage, setItemsPerPage] = useState(20);
    const [selectedEntry, setSelectedEntry] = useState<any | null>(null);
    const [showModal, setShowModal] = useState(false);
    const [userChangedDataType, setUserChangedDataType] = useState(false);
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
                { method:method,provider: provider, model: model ,dataType: dataType },
                { withCredentials: true }
            );

            // Navigate to classification dashboard after successful classification
            if (["sentiment", "legal"].includes((dataType || "").toLowerCase())) {
              navigate(`/datasets/${datasetId}/classifications/${response.data.classification_id}`);
            } else {
              navigate(`/datasets/${datasetId}/classificationsp/${response.data.classification_id}`);
              console.log('uuumedical');
            }


        } catch (err) {
            console.log(err);
            setError(`Classification using ${method.toUpperCase()} failed.`);
        } finally {
            setClassifying(null);
        }
    };
    const handleClassificationandExplanation = async (method: "llm" | "bert") => {
        setClassifying(method);
        try {
            const response = await axios.post(
                `http://localhost:5000/api/classify_and_explain/${datasetId}`,
                { method:method,provider: provider, model: model ,dataType: dataType, limit: classificationLimit},
                { withCredentials: true }
            );

            // Navigate to classification dashboard after successful classification
            if (dataType === "sentiment" || dataType === "legal") {
             navigate(`/datasets/${datasetId}/classifications/${response.data.classification_id}`);
            }
            else if(dataType === "ecqa") {
                 navigate(`/datasets/${datasetId}/classifications_ecqa/${response.data.classification_id}`);
            }
            else if(dataType === "snarks" || dataType === "hotel" ) {
                 navigate(`/datasets/${datasetId}/classifications_snarks/${response.data.classification_id}`);
            }

            else {
              navigate(`/datasets/${datasetId}/classificationsp/${response.data.classification_id}`);
            }


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
                // Only set dataType if user hasn't already changed it (i.e., dataType is still initial).
                if (response.data?.filename && !userChangedDataType) {
                  const lowerName = response.data.filename.toLowerCase();
                  if (
                    lowerName.includes('med') ||
                    lowerName.includes('medical') ||
                    lowerName.includes('pubmed')
                  ) {
                    setDataType('medical');
                  } else if (lowerName.includes('legal') || lowerName.includes('casehold')) {
                    setDataType('legal');

                  }
                  else if (lowerName.includes('bigbench')){
                    setDataType('snarks');
                  }
                  else if (lowerName.includes('cqa')){
                    setDataType('ecqa');
                  }
                  else if (lowerName.includes('deceptive')){
                    setDataType('hotel');
                  }
                  else {
                    setDataType('sentiment');
                  }
                }
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

                <Col md={4} className="border-end border-2">

                    <Card className="border-0">
                    <Button
                        variant="warning"
                        className="mb-2"
                        onClick={handleExploreDataset}
                        disabled={!dataset || exploreLoading}
                    >
                        {exploreLoading ? (
                            <>
                                <Spinner animation="border" size="sm" /> Exploring...
                            </>
                        ) : (
                            "Explore/Annotate Dataset (One by One)"
                        )}
                    </Button>
                        <Card.Body>

                            <Card.Title className="mb-4">Classification Methods</Card.Title>
                            <div className="mb-3 d-flex align-items-center gap-3">
                              <span className="fw-semibold">Select Data Type:</span>
                              <select
                                className="form-select w-auto"
                                value={dataType}
                                onChange={e => {
                                  setDataType(e.target.value as 'sentiment' | 'legal' | 'medical'|'ecqa'|'snarks'|'hotel');
                                  setUserChangedDataType(true);

                                }}
                                style={{ minWidth: 180 }}
                              >
                                <option value="sentiment">Sentiment Analysis</option>
                                <option value="legal">Legal</option>
                                  <option value="medical">Medical</option>
                                  <option value="ecqa">CommonsenseQA</option>
                                  <option value="snarks">Snarks</option>
                                  <option value="hotel">Deceptive Hotel</option>

                              </select>
                            </div>
                            <div className="mb-3 d-flex align-items-center gap-3">
                              <span className="fw-semibold">Entries to Classify:</span>
                              <Form.Control
                                type="number"
                                min="1"
                                max={dataset?.data?.length || 1}
                                value={classificationLimit ?? ""}
                                onChange={e => setClassificationLimit(e.target.value ? Number(e.target.value) : null)}
                                placeholder='5'
                                style={{ width: "100px" }}
                              />
                            </div>
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
                                    variant="primary"
                                    onClick={() => handleClassificationandExplanation("llm")}
                                    disabled={!dataset || !!classifying}
                                >
                                    {classifying === "llm" ? (
                                        <>
                                            <Spinner animation="border" size="sm" /> Classifying...
                                        </>
                                    ) : (
                                        "Classify and explain with LLM"
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
                                <div className="classifications-list" style={{ maxHeight: '400px', overflowY: 'auto', padding: '0 4px' }}>
                                    {classifications.map((classification) => (
                                       <Card
                                          key={classification._id}
                                          className="mb-3 border-0"
                                          onClick={() => {
                                            // Make the check case-insensitive and handle undefined
                                            if (dataType === "sentiment" || dataType === "legal") {
                                              navigate(`/datasets/${datasetId}/classifications/${classification._id}`);
                                            }
                                            else if(dataType === "ecqa") {
                                                 navigate(`/datasets/${datasetId}/classifications_ecqa/${classification._id}`);
                                            }
                                            else if(dataType === "snarks" || dataType==='hotel') {
                                                 navigate(`/datasets/${datasetId}/classifications_snarks/${classification._id}`);
                                            }

                                            else {
                                              navigate(`/datasets/${datasetId}/classificationsp/${classification._id}`);
                                            }
                                          }}
                                          style={{
                                            cursor: 'pointer',
                                            transition: 'all 0.2s ease',
                                            borderRadius: '12px',
                                            boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
                                            borderLeft: `4px solid ${classification.method === "llm" ? '#4361ee' : '#4cc9f0'}`
                                          }}
                                          onMouseEnter={e => e.currentTarget.style.transform = 'translateY(-2px)'}
                                          onMouseLeave={e => e.currentTarget.style.transform = 'translateY(0)'}
                                        >
                                            <Card.Body className="p-4">
                                                <div className="d-flex justify-content-between align-items-start">
                                                    <div style={{ flex: 1 }}>
                                                        <div className="d-flex align-items-center gap-2 mb-3">
                                                            <Badge
                                                                bg={classification.method === "llm" ? "primary" : "info"}
                                                                pill
                                                                style={{
                                                                    fontWeight: 500,
                                                                    background: classification.method === "llm" ? '#4361ee' : '#4cc9f0'
                                                                }}
                                                            >
                                                                {classification.method.toUpperCase()}
                                                            </Badge>
                                                            {getAccuracyBadge(classification.stats.accuracy)}
                                                        </div>

                                                        {classification.method === "llm" && (
                                                            <div className="mb-3">
                                                            <span className="text-muted" style={{ fontSize: '0.85rem' }}>
                                                              Model:
                                                            </span>
                                                                <span className="ms-2 fw-semibold">
                                                              {classification.provider} / {classification.model}
                                                            </span>
                                                            </div>
                                                        )}

                                                        <div className="d-flex align-items-center mt-3">
                                                            <div className="me-4">
                                                                <div className="d-flex align-items-center">
                                                                    <div style={{
                                                                        width: '12px',
                                                                        height: '12px',
                                                                        borderRadius: '50%',
                                                                        background: '#2ecc71',
                                                                        marginRight: '8px'
                                                                    }}></div>
                                                                    <span className="text-muted small">Positive:</span>
                                                                    <span className="ms-2 fw-semibold">{classification.stats.positive}</span>
                                                                </div>
                                                            </div>
                                                            <div>
                                                                <div className="d-flex align-items-center">
                                                                    <div style={{
                                                                        width: '12px',
                                                                        height: '12px',
                                                                        borderRadius: '50%',
                                                                        background: '#e74c3c',
                                                                        marginRight: '8px'
                                                                    }}></div>
                                                                    <span className="text-muted small">Negative:</span>
                                                                    <span className="ms-2 fw-semibold">{classification.stats.negative}</span>
                                                                </div>
                                                            </div>
                                                        </div>
                                                    </div>

                                                    <Button
                                                        variant="outline-danger"
                                                        size="sm"
                                                        onClick={(e) => {
                                                            e.stopPropagation();
                                                            handleDeleteClassification(classification._id, e);
                                                        }}
                                                        style={{
                                                            borderColor: '#ff6b6b',
                                                            color: '#ff6b6b',
                                                            padding: '4px 12px',
                                                            borderRadius: '8px',
                                                            fontWeight: 500
                                                        }}
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
                <Col md={8}>
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
                                        <div style={{ maxHeight: '500px', overflowY: 'auto' }}>
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
                                        </div>
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