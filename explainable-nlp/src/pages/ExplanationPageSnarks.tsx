import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Container, Row, Col, Card, Alert, Spinner, Button, Tab, Tabs, Badge } from 'react-bootstrap';
import axios from 'axios';
import '../index.css';

type AB = '(A)' | '(B)';

function toAB(val: string | number | undefined): AB | undefined {
  if (val === undefined || val === null) return undefined;
  const t = String(val).trim().toUpperCase().replace(/[()\s.]/g, '');
  if (t === 'A') return '(A)';
  if (t === 'B') return '(B)';
  return undefined;
}

interface SnarksEntry {
  question: string;
  label: string;
  actualLabel?: string;
  score: number;
  method?: string;
  llm_explanations?: Record<string, string>;
  shapwithllm_explanations?: Record<string, string>;
  ratings?: Record<string, { llm?: number; combined?: number }>;
  // faithfulness-only metrics (and subs)
  faithfulness_score?: number;
  qag_score?: number;
  counterfactual?: number;
  contextual_faithfulness?: number;
}

interface ModelInfo {
  provider: string;
  model: string;
  id: string;
}

interface ExplanationData {
  llm?: string;
  combined?: string | null;
}

const ExplanationPageSnarks: React.FC = () => {
  const { datasetId, classificationId, resultId } = useParams();
  const navigate = useNavigate();

  const [entry, setEntry] = useState<SnarksEntry | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [totalResults, setTotalResults] = useState(0);
  const [currentResultIndex, setCurrentResultIndex] = useState(0);

  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([]);
  const [activeModel, setActiveModel] = useState<string>('');
  const [explanations, setExplanations] = useState<Record<string, ExplanationData>>({});
  const [ratings, setRatings] = useState<Record<string, Record<string, number>>>({});
  const [isExplaining, setIsExplaining] = useState(false);
  const [isSubmittingRatings, setIsSubmittingRatings] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      setError(null);
      try {
        const [entryRes, classRes] = await Promise.all([
          axios.get(`http://localhost:5000/api/classificationentry/${classificationId}/${resultId}`, { withCredentials: true }),
          axios.get(`http://localhost:5000/api/classification/${classificationId}`, { withCredentials: true }),
        ]);

        const e: SnarksEntry = entryRes.data;
        setEntry(e);
        setTotalResults(classRes.data.results?.length || 0);
        setCurrentResultIndex(Number(resultId) || 0);

        const savedModels = classRes.data.explanation_models || [];
        const initialData: Record<string, ExplanationData> = {};
        const initialRatings: Record<string, Record<string, number>> = {};

        savedModels.forEach((m: any) => {
          const modelId = `${m.provider}-${m.model}`.toLowerCase();
          initialData[modelId] = {
            llm: e.llm_explanations?.[m.model],
            combined: e.shapwithllm_explanations?.[m.model] ?? null,
          };
          initialRatings[modelId] = {
            llm: e.ratings?.[modelId]?.llm || 0,
            combined: e.ratings?.[modelId]?.combined || 0,
          };
        });

        setAvailableModels(savedModels.map((m: any) => ({
          id: `${m.provider}-${m.model}`.toLowerCase(),
          provider: m.provider,
          model: m.model,
        })));

        setExplanations(initialData);
        setRatings(initialRatings);
        setActiveModel(Object.keys(initialData)[0] || '');
      } catch (err) {
        setError('Failed to load data');
      }
      setLoading(false);
    };
    fetchData();
  }, [classificationId, resultId]);

  const generateLLMExplanation = async (modelId: string) => {
    setIsExplaining(true);
    const model = availableModels.find(m => m.id === modelId);
    if (!model) return setIsExplaining(false);

    try {
      const llmResponse = await axios.post('http://localhost:5000/api/explain', {
        text: entry?.question,
        provider: model.provider,
        model: model.model,
        explainer_type: 'llm',
        resultId,
        predictedlabel: entry?.label,
        confidence: entry?.score,
        truelabel: entry?.actualLabel,
        classificationId,
        datatype: "snarks"
      }, { withCredentials: true });

      setExplanations(prev => ({
        ...prev,
        [modelId]: {
          llm: llmResponse.data.explanation,
          combined: null,
        }
      }));
    } catch {
      setError('Failed to generate explanation');
    } finally {
      setIsExplaining(false);
    }
  };

  const handleRatingChange = (modelId: string, type: string, rating: number) => {
    setRatings(prev => ({
      ...prev,
      [modelId]: {
        ...prev[modelId],
        [type]: rating
      }
    }));
  };

  const submitRatings = async () => {
    setIsSubmittingRatings(true);
    try {
      await axios.post(
        'http://localhost:5000/api/save_ratings',
        { classificationId, resultId, ratings, timestamp: new Date().toISOString() },
        { withCredentials: true }
      );
      alert('Ratings submitted successfully!');
    } catch {
      setError('Failed to submit ratings');
    } finally {
      setIsSubmittingRatings(false);
    }
  };

  const hasRatings = () =>
    Object.values(ratings).some(modelRatings => Object.values(modelRatings).some(v => v > 0));

  const handlePrevious = () => {
    const newIndex = currentResultIndex - 1;
    navigate(`/datasets/${datasetId}/classifications_snarks/${classificationId}/results/${newIndex}`);
  };
  const handleNext = () => {
    const newIndex = currentResultIndex + 1;
    navigate(`/datasets/${datasetId}/classifications_snarks/${classificationId}/results/${newIndex}`);
  };

  if (loading) {
    return (
      <Container className="py-5 text-center">
        <Spinner animation="border" />
        <div className="mt-3">Loading Snarks entry...</div>
      </Container>
    );
  }
  if (error) {
    return (
      <Container className="py-5">
        <Alert variant="danger">{error}</Alert>
        <Button onClick={() => navigate(-1)}>Back</Button>
      </Container>
    );
  }
  if (!entry) return null;

  const predAB = toAB(entry.label);
  const goldAB = toAB(entry.actualLabel);

  return (
    <Container className="py-4 explanation-page" fluid>
      <div className="d-flex justify-content-between align-items-center mb-4">
        <Button
          variant="outline-secondary"
          onClick={() => navigate(`/datasets/${datasetId}/classifications_snarks/${classificationId}`)}
        >
          ← Back to Snarks Classification
        </Button>
        <div className="d-flex align-items-center gap-3">
          <div className="text-muted">
            Result {currentResultIndex + 1} of {totalResults}
          </div>
          <div className="d-flex gap-2">
            <Button variant="outline-primary" onClick={handlePrevious} disabled={currentResultIndex === 0}>
              ← Previous
            </Button>
            <Button
              variant="outline-primary"
              onClick={handleNext}
              disabled={currentResultIndex >= totalResults - 1}
            >
              Next →
            </Button>
          </div>
        </div>
      </div>

      {/* MAIN CARD: Question, Prediction, Actual */}
      <Card className="mb-4">
        <Card.Body>
          <Row>
            <Col md={8}>
              <h5>Question</h5>
              <div className="p-3 bg-light rounded mb-3" style={{ whiteSpace: 'pre-wrap' }}>
                {entry.question}
              </div>
              <div className="text-muted small">
                Confidence: {(entry.score * 100).toFixed(1)}%
              </div>
            </Col>
            <Col md={4}>
              <div className="d-flex flex-column gap-3">
                <div className="text-center">
                  <div className="text-muted small">Prediction</div>
                  <Badge pill bg={predAB === '(A)' ? 'primary' : 'warning'} className="px-3 py-2 fs-6">
                    {predAB || entry.label}
                  </Badge>
                </div>
                <div className="text-center">
                  <div className="text-muted small">Actual Label</div>
                  <Badge pill bg="success" className="px-3 py-2 fs-6">
                    {goldAB || "N/A"}
                  </Badge>
                </div>
              </div>
            </Col>
          </Row>
        </Card.Body>
      </Card>

      {/* Faithfulness Metrics (Snarks-only) */}
      <Card className="mb-4">
        <Card.Body>
          <Card.Title>Faithfulness Metrics</Card.Title>
          <Row className="mt-2 g-3">
            <Col md={3}>
              <div className="d-flex flex-column">
                <span className="text-muted small">Faithfulness</span>
                <Badge bg="secondary" className="fs-6 align-self-start mt-1">
                  {entry?.faithfulness_score !== undefined && entry?.faithfulness_score !== null
                    ? Number(entry.faithfulness_score).toFixed(2)
                    : 'N/A'}
                </Badge>
              </div>
            </Col>
            <Col md={3}>
              <div className="d-flex flex-column">
                <span className="text-muted small">QAG</span>
                <Badge bg="info" className="fs-6 align-self-start mt-1">
                  {entry?.qag_score !== undefined && entry?.qag_score !== null
                    ? Number(entry.qag_score).toFixed(2)
                    : 'N/A'}
                </Badge>
              </div>
            </Col>
            <Col md={3}>
              <div className="d-flex flex-column">
                <span className="text-muted small">Counterfactual</span>
                <Badge bg="warning" text="dark" className="fs-6 align-self-start mt-1">
                  {entry?.counterfactual !== undefined && entry?.counterfactual !== null
                    ? Number(entry.counterfactual).toFixed(2)
                    : 'N/A'}
                </Badge>
              </div>
            </Col>
            <Col md={3}>
              <div className="d-flex flex-column">
                <span className="text-muted small">Contextual Faithfulness</span>
                <Badge bg="primary" className="fs-6 align-self-start mt-1">
                  {entry?.contextual_faithfulness !== undefined && entry?.contextual_faithfulness !== null
                    ? Number(entry.contextual_faithfulness).toFixed(2)
                    : 'N/A'}
                </Badge>
              </div>
            </Col>
          </Row>
        </Card.Body>
      </Card>

      {/* LLM Explanations */}
      <Card className="h-100">
        <Card.Header>
          <div className="d-flex justify-content-between align-items-center">
            <Card.Title className="mb-0">LLM Explanations</Card.Title>
            <div className="d-flex gap-2">
              <Button
                size="sm"
                variant="outline-primary"
                onClick={() => generateLLMExplanation(activeModel)}
                disabled={isExplaining || !activeModel}
              >
                {isExplaining ? (<Spinner size="sm" className="me-2" />) : null}
                Generate Current
              </Button>
            </div>
          </div>
        </Card.Header>
        <Card.Body className="p-0">
          <Tabs
            activeKey={activeModel}
            onSelect={(k) => setActiveModel(k as string)}
            className="model-tabs border-bottom-0"
            fill
          >
            {availableModels.map(model => (
              <Tab
                key={model.id}
                eventKey={model.id}
                title={<div className="d-flex align-items-center justify-content-center gap-2">{model.model}</div>}
              >
                <div className="p-4">
                  <div className="explanation-section">
                    <h6 className="text-primary mb-3">Direct Explanation</h6>
                    <div className="explanation-content mb-3">
                      {explanations[model.id]?.llm ? (
                        <div className="p-3 bg-light rounded" style={{ whiteSpace: 'pre-wrap' }}>
                          {explanations[model.id].llm}
                        </div>
                      ) : (
                        <div className="text-muted text-center py-4 border rounded">
                          No explanation generated yet
                        </div>
                      )}
                    </div>

                    <RatingSection
                      title="Direct Explanation"
                      value={ratings[model.id]?.llm || 0}
                      onChange={(rating: number) => handleRatingChange(model.id, 'llm', rating)}
                      disabled={!explanations[model.id]?.llm}
                    />
                  </div>
                </div>
              </Tab>
            ))}
          </Tabs>
        </Card.Body>
      </Card>

      <div className="d-flex justify-content-end mt-4">
        <Button
          variant="success"
          size="lg"
          onClick={submitRatings}
          disabled={isSubmittingRatings || !hasRatings()}
          className="submit-ratings-btn"
        >
          {isSubmittingRatings ? (<Spinner size="sm" className="me-2" />) : null}
          Submit All Ratings
        </Button>
      </div>
    </Container>
  );
};

interface RatingSectionProps {
  title: string;
  value: number;
  onChange: (rating: number) => void;
  disabled: boolean;
}

const RatingSection: React.FC<RatingSectionProps> = ({ title, value, onChange, disabled }) => (
  <div className="rating-section">
    <div className="d-flex justify-content-between align-items-center">
      <span className="small text-muted">Rate {title}:</span>
      <div className="d-flex gap-1">
        {[1, 2, 3, 4, 5].map((rating) => (
          <button
            key={rating}
            className={`rating-star ${value >= rating ? 'active' : ''} ${disabled ? 'disabled' : ''}`}
            onClick={() => !disabled && onChange(rating)}
            disabled={disabled}
            style={{
              minWidth: '30px',
              height: '30px',
              border: '1px solid #dee2e6',
              borderRadius: '4px',
              backgroundColor: value >= rating ? '#007bff' : 'white',
              color: value >= rating ? 'white' : '#6c757d',
              cursor: disabled ? 'not-allowed' : 'pointer',
              fontSize: '14px',
              fontWeight: '500',
              opacity: disabled ? 0.5 : 1
            }}
          >
            {rating}
          </button>
        ))}
      </div>
    </div>
    {value > 0 && (
      <div className="text-end small mt-1">
        <span className="text-muted">Your rating:</span> {value}/5
      </div>
    )}
  </div>
);

export default ExplanationPageSnarks;