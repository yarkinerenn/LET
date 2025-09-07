import React, { useEffect, useState } from "react";
import { useParams, Link, useNavigate } from "react-router-dom";
import { Container, Row, Col, Table, Button, Alert, Spinner, Card, Badge, Pagination, Form, Modal } from "react-bootstrap";
import axios from "axios";
// Using Bootstrap Icons instead of lucide-react

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
            } else if (["legal"].includes((response.data.data_type || dataType).toLowerCase())) {
                navigate(`/datasets/${datasetId}/classifications/${response.data.classification_id}/results/0`);
            }
            else if (["legal"].includes((response.data.data_type || dataType).toLowerCase())) {
                navigate(`/datasets/${datasetId}/classifications_legal/${response.data.classification_id}/results/0`);
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
                `http://localhost:5000/api/classify_only/${datasetId}`,
                { method:method,provider: provider, model: model ,dataType: dataType, limit: classificationLimit },
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
    
    const handleClassificationBERT = async () => {
        setClassifying("bert");
        try {
            const response = await axios.post(
                `http://localhost:5000/api/classify/${datasetId}`,
                { method: "bert", provider: provider, model: model, dataType: dataType, limit: classificationLimit },
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
            setError(`Classification using BERT failed.`);
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
            if (dataType === "sentiment") {
             navigate(`/datasets/${datasetId}/classifications/${response.data.classification_id}`);
            }
            else if(dataType === "ecqa") {
                 navigate(`/datasets/${datasetId}/classifications_ecqa/${response.data.classification_id}`);
            }
            else if(dataType === "snarks") {
                 navigate(`/datasets/${datasetId}/classifications_snarks/${response.data.classification_id}`);
            }
            else if( dataType === "hotel" ) {
                 navigate(`/datasets/${datasetId}/classifications_hotel/${response.data.classification_id}`);
            }
            else if(dataType === "legal") {
                 navigate(`/datasets/${datasetId}/classifications_legal/${response.data.classification_id}`);
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
        <div className="dataset-view-container" style={{ 
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            minHeight: '100vh',
            padding: '2rem 0'
        }}>
            <Container fluid style={{ maxWidth: '1400px' }}>
                {/* Header Section */}
                <Row className="mb-4">
                    <Col>
                        <div className="d-flex align-items-center justify-content-between">
                            <div className="d-flex align-items-center">
                                <Button
                                    variant="light"
                                    size="lg"
                                    onClick={() => navigate(`/datasets`)}
                                    className="me-3 rounded-pill px-4"
                                    style={{ 
                                        background: 'rgba(255,255,255,0.2)', 
                                        border: 'none',
                                        color: 'white',
                                        backdropFilter: 'blur(10px)'
                                    }}
                                >
                                    <i className="bi bi-arrow-left me-2"></i>
                                    Back to Datasets
                                </Button>
                                <div>
                                    <h1 className="text-white mb-1" style={{ fontSize: '2.5rem', fontWeight: '700' }}>
                                        {dataset?.filename || 'Dataset View'}
                                    </h1>
                                    <p className="text-white-50 mb-0" style={{ fontSize: '1.1rem' }}>
                                        Analyze and classify your dataset with AI
                                    </p>
                                </div>
                            </div>
                            <div className="text-end">
                                <div className="text-white-50 mb-1">Total Entries</div>
                                <div className="text-white h3 mb-0">{dataset?.data?.length || 0}</div>
                            </div>
                        </div>
                    </Col>
                </Row>

                <Row className="g-4">
                {/* Classification Sidebar */}
                    <Col lg={4}>
                        <div className="sticky-top" style={{ top: '2rem' }}>
                            {/* Explore Dataset Card */}
                            <Card className="mb-4 border-0 shadow-lg" style={{ 
                                background: 'rgba(255,255,255,0.95)',
                                backdropFilter: 'blur(20px)',
                                borderRadius: '20px'
                            }}>
                                <Card.Body className="p-4">
                                    <div className="d-flex align-items-center mb-3">
                                        <div className="p-2 rounded-circle me-3" style={{ background: 'linear-gradient(45deg, #ff6b6b, #ffa500)' }}>
                                            <i className="bi bi-eye-fill text-white" style={{ fontSize: '20px' }}></i>
                                        </div>
                                        <h5 className="mb-0 fw-bold">Explore Dataset</h5>
                                    </div>
                                    <p className="text-muted mb-3">Review and annotate entries one by one</p>
                    <Button
                        variant="warning"
                                        size="lg"
                        onClick={handleExploreDataset}
                        disabled={!dataset || exploreLoading}
                                        className="w-100 rounded-pill py-3"
                                        style={{ 
                                            background: 'linear-gradient(45deg, #ff6b6b, #ffa500)',
                                            border: 'none',
                                            fontWeight: '600',
                                            fontSize: '1.1rem'
                                        }}
                    >
                        {exploreLoading ? (
                            <>
                                                <Spinner animation="border" size="sm" className="me-2" />
                                                Exploring...
                            </>
                        ) : (
                                            <>
                                                <i className="bi bi-play-fill me-2"></i>
                                                Start Exploration
                                            </>
                        )}
                    </Button>
                                </Card.Body>
                            </Card>
                            {/* Classification Methods Card */}
                            <Card className="mb-4 border-0 shadow-lg" style={{ 
                                background: 'rgba(255,255,255,0.95)',
                                backdropFilter: 'blur(20px)',
                                borderRadius: '20px'
                            }}>
                                <Card.Body className="p-4">
                                    <div className="d-flex align-items-center mb-4">
                                        <div className="p-2 rounded-circle me-3" style={{ background: 'linear-gradient(45deg, #667eea, #764ba2)' }}>
                                            <i className="bi bi-cpu-fill text-white" style={{ fontSize: '20px' }}></i>
                                        </div>
                                        <h5 className="mb-0 fw-bold">AI Classification</h5>
                        </div>
                                    
                                    <div className="mb-4">
                                        <label className="form-label fw-semibold text-muted mb-2">
                                            <i className="bi bi-bullseye me-2"></i>
                                            Entries to Classify
                                        </label>
                              <Form.Control
                                type="number"
                                min="1"
                                max={dataset?.data?.length || 1}
                                value={classificationLimit ?? ""}
                                onChange={e => setClassificationLimit(e.target.value ? Number(e.target.value) : null)}
                                            placeholder="5"
                                            className="rounded-pill"
                                            style={{ 
                                                border: '2px solid #e9ecef',
                                                padding: '0.75rem 1rem'
                                            }}
                              />
                            </div>

                            <div className="d-grid gap-3">
                                <Button
                                    variant="primary"
                                            size="lg"
                                    onClick={() => handleClassification("llm")}
                                    disabled={!dataset || !!classifying}
                                            className="rounded-pill py-3"
                                            style={{ 
                                                background: 'linear-gradient(45deg, #667eea, #764ba2)',
                                                border: 'none',
                                                fontWeight: '600'
                                            }}
                                >
                                    {classifying === "llm" ? (
                                        <>
                                                    <Spinner animation="border" size="sm" className="me-2" />
                                                    Classifying...
                                        </>
                                    ) : (
                                                <>
                                                    <i className="bi bi-lightning-fill me-2"></i>
                                                    Classify with LLM
                                                </>
                                    )}
                                </Button>
                                        
                                <Button
                                            variant="info"
                                            size="lg"
                                    onClick={() => handleClassificationandExplanation("llm")}
                                    disabled={!dataset || !!classifying}
                                            className="rounded-pill py-3"
                                            style={{ 
                                                background: 'linear-gradient(45deg, #4facfe, #00f2fe)',
                                                border: 'none',
                                                fontWeight: '600'
                                            }}
                                >
                                    {classifying === "llm" ? (
                                        <>
                                                    <Spinner animation="border" size="sm" className="me-2" />
                                                    Classifying...
                                        </>
                                    ) : (
                                                <>
                                                    <i className="bi bi-file-text-fill me-2"></i>
                                                    Classify & Explain
                                                </>
                                    )}
                                </Button>
                                        
                                {dataType === "sentiment" && (
                                    <Button
                                        variant="success"
                                                size="lg"
                                        onClick={handleClassificationBERT}
                                        disabled={!dataset || !!classifying}
                                                className="rounded-pill py-3"
                                                style={{ 
                                                    background: 'linear-gradient(45deg, #56ab2f, #a8e6cf)',
                                                    border: 'none',
                                                    fontWeight: '600'
                                                }}
                                    >
                                        {classifying === "bert" ? (
                                            <>
                                                        <Spinner animation="border" size="sm" className="me-2" />
                                                        Classifying...
                                            </>
                                        ) : (
                                                    <>
                                                        <i className="bi bi-database-fill me-2"></i>
                                                        Classify with BERT
                                                    </>
                                        )}
                                    </Button>
                                )}
                            </div>
                        </Card.Body>
                    </Card>
                            {/* Previous Classifications Card */}
                            <Card className="border-0 shadow-lg" style={{ 
                                background: 'rgba(255,255,255,0.95)',
                                backdropFilter: 'blur(20px)',
                                borderRadius: '20px'
                            }}>
                                <Card.Body className="p-4">
                                    <div className="d-flex align-items-center mb-4">
                                        <div className="p-2 rounded-circle me-3" style={{ background: 'linear-gradient(45deg, #ff9a9e, #fecfef)' }}>
                                            <i className="bi bi-bar-chart-fill text-white" style={{ fontSize: '20px' }}></i>
                                        </div>
                                        <h5 className="mb-0 fw-bold">Previous Classifications</h5>
                                    </div>
                                    
                            {loadingClassifications ? (
                                        <div className="text-center py-4">
                                    <Spinner animation="border" size="sm" />
                                            <p className="text-muted mt-2 mb-0">Loading classifications...</p>
                                </div>
                            ) : classifications.length > 0 ? (
                                        <div className="classifications-list" style={{ maxHeight: '500px', overflowY: 'auto' }}>
                                    {classifications
                                      .filter(c => c.method !== "explore")
                                      .map((classification) => (
                                       <Card
                                          key={classification._id}
                                          className="mb-3 border-0"
                                          onClick={() => {
                                            if (dataType === "sentiment") {
                                              navigate(`/datasets/${datasetId}/classifications/${classification._id}`);
                                            }
                                            else if(dataType === "ecqa") {
                                                 navigate(`/datasets/${datasetId}/classifications_ecqa/${classification._id}`);
                                            }
                                            else if(dataType === "legal") {
                                                navigate(`/datasets/${datasetId}/classifications_legal/${classification._id}`);
                                           }
                                            else if(dataType === "snarks") {
                                                 navigate(`/datasets/${datasetId}/classifications_snarks/${classification._id}`);
                                            }
                                            else if(dataType === 'hotel') {
                                                 navigate(`/datasets/${datasetId}/classifications_hotel/${classification._id}`);
                                            }
                                            else {
                                              navigate(`/datasets/${datasetId}/classificationsp/${classification._id}`);
                                            }
                                          }}
                                          style={{
                                            cursor: 'pointer',
                                                    transition: 'all 0.3s ease',
                                                    borderRadius: '16px',
                                                    background: 'rgba(255,255,255,0.8)',
                                                    border: '1px solid rgba(0,0,0,0.05)',
                                                    boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
                                                  }}
                                                  onMouseEnter={e => {
                                                    e.currentTarget.style.transform = 'translateY(-4px)';
                                                    e.currentTarget.style.boxShadow = '0 8px 25px rgba(0,0,0,0.15)';
                                                  }}
                                                  onMouseLeave={e => {
                                                    e.currentTarget.style.transform = 'translateY(0)';
                                                    e.currentTarget.style.boxShadow = '0 4px 12px rgba(0,0,0,0.1)';
                                                  }}
                                                >
                                                    <Card.Body className="p-3">
                                                <div className="d-flex justify-content-between align-items-start">
                                                    <div style={{ flex: 1 }}>
                                                                <div className="d-flex align-items-center gap-2 mb-2">
                                                            <Badge
                                                                        className="rounded-pill px-3 py-2"
                                                                style={{
                                                                            background: classification.method === "llm" 
                                                                                ? 'linear-gradient(45deg, #667eea, #764ba2)' 
                                                                                : 'linear-gradient(45deg, #4facfe, #00f2fe)',
                                                                            border: 'none',
                                                                            fontWeight: '600',
                                                                            fontSize: '0.8rem'
                                                                        }}
                                                                    >
                                                                        {classification.method === "llm" ? "🤖 LLM" : "🧠 BERT"}
                                                            </Badge>
                                                            {getAccuracyBadge(classification.stats.accuracy)}
                                                        </div>

                                                        {classification.method === "llm" && (
                                                                    <div className="mb-2">
                                                                        <div className="d-flex align-items-center text-muted" style={{ fontSize: '0.85rem' }}>
                                                                            <i className="bi bi-gear-fill me-1" style={{ fontSize: '14px' }}></i>
                                                                            <span className="fw-semibold">
                                                              {classification.provider} / {classification.model}
                                                            </span>
                                                                        </div>
                                                            </div>
                                                        )}

                                                                <div className="d-flex align-items-center text-muted" style={{ fontSize: '0.8rem' }}>
                                                                    <i className="bi bi-clock-fill me-1" style={{ fontSize: '12px' }}></i>
                                                                    <span>{formatDate(classification.created_at)}</span>
                                                                </div>
                                                    </div>

                                                    <Button
                                                        variant="outline-danger"
                                                        size="sm"
                                                        onClick={(e) => {
                                                            e.stopPropagation();
                                                            handleDeleteClassification(classification._id, e);
                                                        }}
                                                                className="rounded-pill px-3"
                                                        style={{
                                                            borderColor: '#ff6b6b',
                                                            color: '#ff6b6b',
                                                                    fontWeight: '500',
                                                                    fontSize: '0.8rem'
                                                        }}
                                                    >
                                                                <i className="bi bi-trash-fill" style={{ fontSize: '14px' }}></i>
                                                    </Button>
                                                </div>
                                            </Card.Body>
                                        </Card>
                                    ))}
                                </div>
                            ) : (
                                        <div className="text-center py-4">
                                            <div className="p-3 rounded-circle d-inline-block mb-3" style={{ background: 'rgba(0,0,0,0.05)' }}>
                                                <i className="bi bi-bar-chart text-muted" style={{ fontSize: '24px' }}></i>
                                            </div>
                                            <p className="text-muted mb-0">No previous classifications</p>
                                            <small className="text-muted">Start by classifying your dataset above</small>
                                        </div>
                            )}
                        </Card.Body>
                    </Card>
                        </div>
                </Col>

                {/* Main Content */}
                    <Col lg={8}>
                    {loading ? (
                            <div className="text-center py-5">
                                <div className="p-4 rounded-circle d-inline-block mb-3" style={{ background: 'rgba(255,255,255,0.1)' }}>
                                    <Spinner animation="border" style={{ color: 'white' }} />
                                </div>
                                <p className="text-white-50">Loading dataset...</p>
                        </div>
                    ) : error ? (
                            <Alert variant="danger" className="border-0 shadow-lg" style={{ borderRadius: '16px' }}>
                                <div className="d-flex align-items-center">
                                    <i className="bi bi-x-circle-fill me-2"></i>
                                    {error}
                                </div>
                            </Alert>
                        ) : (
                            <Card className="border-0 shadow-lg" style={{ 
                                background: 'rgba(255,255,255,0.95)',
                                backdropFilter: 'blur(20px)',
                                borderRadius: '20px'
                            }}>
                                <Card.Body className="p-4">
                                    <div className="d-flex justify-content-between align-items-center mb-4">
                                        <div>
                                            <h4 className="mb-1 fw-bold">Dataset Preview</h4>
                                            <p className="text-muted mb-0">Browse and explore your data entries</p>
                                        </div>
                                        <div className="d-flex align-items-center">
                                            <span className="text-muted me-3">Items per page:</span>
                                    <Form.Select 
                                        size="sm" 
                                                style={{ width: '120px' }}
                                        value={itemsPerPage}
                                        onChange={handleItemsPerPageChange}
                                                className="rounded-pill"
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
                                            <div style={{ 
                                                maxHeight: '600px', 
                                                overflowY: 'auto',
                                                borderRadius: '12px',
                                                border: '1px solid rgba(0,0,0,0.05)'
                                            }}>
                                            <Table hover responsive className="mb-0">
                                                    <thead style={{ 
                                                        background: 'linear-gradient(45deg, #f8f9fa, #e9ecef)',
                                                        position: 'sticky',
                                                        top: 0,
                                                        zIndex: 10
                                                    }}>
                                                <tr>
                                                    {Object.keys(dataset.data[0]).map((key) => (
                                                            <th key={key} className="px-4 py-3 fw-semibold text-dark border-0">
                                                                {key}
                                                            </th>
                                                    ))}
                                                </tr>
                                                </thead>
                                                <tbody>
                                                {currentItems.map((row, index) => (
                                                    <tr
                                                        key={index}
                                                        onClick={() => handleEntryClick(row)}
                                                            style={{ 
                                                                cursor: 'pointer',
                                                                transition: 'all 0.2s ease'
                                                            }}
                                                        className="hover-highlight"
                                                            onMouseEnter={e => {
                                                                e.currentTarget.style.background = 'rgba(102, 126, 234, 0.05)';
                                                            }}
                                                            onMouseLeave={e => {
                                                                e.currentTarget.style.background = '';
                                                            }}
                                                    >
                                                        {Object.values(row).map((value, i) => (
                                                                <td key={i} className="px-4 py-3 text-truncate border-0" style={{ maxWidth: '300px' }}>
                                                                    <span className="text-dark">{String(value)}</span>
                                                            </td>
                                                        ))}
                                                    </tr>
                                                ))}
                                                </tbody>
                                            </Table>
                                        </div>

                                            <div className="d-flex justify-content-between align-items-center mt-4">
                                                <div className="text-muted">
                                                    <i className="bi bi-database me-2"></i>
                                                    Showing {indexOfFirstItem + 1} to {Math.min(indexOfLastItem, dataset.data.length)} of {dataset.data.length} entries
                                                </div>
                                        <Pagination className="mb-0">
                                                    <Pagination.First 
                                                        onClick={() => setCurrentPage(1)} 
                                                        disabled={currentPage === 1}
                                                        className="rounded-pill"
                                                    />
                                                    <Pagination.Prev 
                                                        onClick={() => setCurrentPage(curr => Math.max(curr - 1, 1))} 
                                                        disabled={currentPage === 1}
                                                        className="rounded-pill"
                                                    />
                                            {paginationItems}
                                                    <Pagination.Next 
                                                        onClick={() => setCurrentPage(curr => Math.min(curr + 1, totalPages))} 
                                                        disabled={currentPage === totalPages}
                                                        className="rounded-pill"
                                                    />
                                                    <Pagination.Last 
                                                        onClick={() => setCurrentPage(totalPages)} 
                                                        disabled={currentPage === totalPages}
                                                        className="rounded-pill"
                                                    />
                                        </Pagination>
                                    </div>
                                </>
                            ) : (
                                        <div className="text-center py-5">
                                            <div className="p-4 rounded-circle d-inline-block mb-3" style={{ background: 'rgba(0,0,0,0.05)' }}>
                                                <i className="bi bi-database text-muted" style={{ fontSize: '32px' }}></i>
                                            </div>
                                            <h5 className="text-muted">No data available</h5>
                                            <p className="text-muted">This dataset appears to be empty</p>
                                        </div>
                                    )}
                                </Card.Body>
                            </Card>
                    )}
                </Col>
            </Row>
            </Container>

            {/* Modal for viewing full entry details */}
            <Modal show={showModal} onHide={() => setShowModal(false)} size="lg" centered>
                <Modal.Header closeButton className="border-0" style={{ 
                    background: 'linear-gradient(45deg, #667eea, #764ba2)',
                    color: 'white',
                    borderRadius: '16px 16px 0 0'
                }}>
                    <Modal.Title className="d-flex align-items-center">
                        <i className="bi bi-file-text-fill me-2"></i>
                        Entry Details
                    </Modal.Title>
                </Modal.Header>
                <Modal.Body className="p-4">
                    {selectedEntry && (
                        <div>
                            {Object.entries(selectedEntry).map(([key, value]) => (
                                <div key={key} className="mb-4">
                                    <h6 className="text-muted mb-3 fw-semibold d-flex align-items-center">
                                        <i className="bi bi-database me-2"></i>
                                        {key}
                                    </h6>
                                    <div 
                                        style={{ 
                                            whiteSpace: 'pre-wrap',
                                            background: 'linear-gradient(45deg, #f8f9fa, #e9ecef)',
                                            border: '1px solid rgba(0,0,0,0.05)',
                                            borderRadius: '12px'
                                        }} 
                                        className="p-4"
                                    >
                                        <span className="text-dark">{String(value)}</span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </Modal.Body>
                <Modal.Footer className="border-0">
                    <Button 
                        variant="secondary" 
                        onClick={() => setShowModal(false)}
                        className="rounded-pill px-4"
                    >
                        Close
                    </Button>
                </Modal.Footer>
            </Modal>
        </div>
    );
};

// Add some custom styles
const styles = `
.hover-highlight:hover {
    background-color: rgba(102, 126, 234, 0.1) !important;
    transform: translateY(-1px);
}

.dataset-view-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

/* Custom scrollbar */
.classifications-list::-webkit-scrollbar {
    width: 6px;
}

.classifications-list::-webkit-scrollbar-track {
    background: rgba(0,0,0,0.05);
    border-radius: 10px;
}

.classifications-list::-webkit-scrollbar-thumb {
    background: rgba(102, 126, 234, 0.3);
    border-radius: 10px;
}

.classifications-list::-webkit-scrollbar-thumb:hover {
    background: rgba(102, 126, 234, 0.5);
}

/* Button hover effects */
.btn {
    transition: all 0.3s ease;
}

.btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
}

/* Card hover effects */
.card {
    transition: all 0.3s ease;
}

/* Pagination styling */
.pagination .page-link {
    border-radius: 8px !important;
    margin: 0 2px;
    border: none;
    color: #667eea;
}

.pagination .page-link:hover {
    background-color: #667eea;
    color: white;
    transform: translateY(-1px);
}

.pagination .page-item.active .page-link {
    background-color: #667eea;
    border-color: #667eea;
}

/* Form control styling */
.form-control:focus {
    border-color: #667eea;
    box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25);
}

/* Badge styling */
.badge {
    font-weight: 600;
    letter-spacing: 0.5px;
}

/* Table styling */
.table th {
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    font-size: 0.85rem;
}

/* Modal styling */
.modal-content {
    border-radius: 16px;
    border: none;
    box-shadow: 0 20px 60px rgba(0,0,0,0.2);
}

/* Loading spinner */
.spinner-border {
    width: 1.5rem;
    height: 1.5rem;
}
`;

// Add styles to document
const styleSheet = document.createElement("style");
styleSheet.innerText = styles;
document.head.appendChild(styleSheet);

export default DatasetView;