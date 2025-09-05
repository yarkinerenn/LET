import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { Navbar, Container, Nav, Button } from 'react-bootstrap';
import { AuthProvider, useAuth } from "./modules/auth";
import { Login } from "./pages/login";
import Register from "./pages/register";
import 'bootstrap/dist/css/bootstrap.min.css';
import Dashboard from "./pages/dashboard";
import HomePage from "./pages/Homepage";
import Settings from "./pages/settings";
import Datasets from "./pages/datasets";
import 'bootstrap/dist/css/bootstrap.min.css';
import 'bootstrap-icons/font/bootstrap-icons.css';
import DatasetView from "./pages/Datasetview";
import PrivateRoute from "./modules/PrivateRoute";
import {ProviderContextProvider} from "./modules/provider";
import ClassificationDashboard from "./pages/ClassificationDashboard";
import ExplanationPage from "./pages/ExplanationPage";
import ExplanationPageDashboard from "./pages/ExplanationPageDashboard";
import ClassificationDashboardPubMedQA from "./pages/ClassificationDashboardPubMedQA";
import ExplanationPagePubMedQA from "./pages/ExplanationPagePub";
import ECQADashboard from "./pages/ECQADashboard";
import ExplanationPageECQA from "./pages/ExplanationPageECQA";
import SnarksDashboard from "./pages/SnarksDashboard";
import ExplanationPageSnarks from "./pages/ExplanationPageSnarks";
import ExplanationPageHotel from "./pages/ExplanationPageHotel";
import HotelDashboard from "./pages/HotelDashboard";
import SentimentDashboard from './pages/ClassificationDashboardSentiment';
function AppContent() {
    const { user, logout } = useAuth();

    return (
        <>
            <Navbar bg="dark" variant="dark" expand="lg" className="shadow-sm">
                <Container>
                    <Navbar.Brand as={Link} to="/" className="fw-bold">
                        <i className="bi bi-shield-lock me-2"></i>
                        XNLP
                    </Navbar.Brand>
                    <Navbar.Toggle aria-controls="basic-navbar-nav" />
                    <Navbar.Collapse id="basic-navbar-nav">
                        <Nav className="ms-auto align-items-center">
                            {!user ? (
                                <>
                                    <Nav.Link as={Link} to="/login" className="text-light">
                                        Login
                                    </Nav.Link>
                                    <Nav.Link as={Link} to="/register" className="text-light">
                                        Register
                                    </Nav.Link>
                                </>
                            ) : (
                                <>
                                    <Nav.Link as={Link} to="/datasets" className="text-light fw-bold">
                                        My Datasets
                                    </Nav.Link>

                                    <Nav.Item className="ms-3">
                                        <Link to="/settings" className="text-light">
                                            <i className="bi bi-gear fs-5"></i>
                                        </Link>
                                    </Nav.Item>

                                    <Button variant="outline-light" onClick={logout} className="ms-3">
                                        Logout
                                    </Button>
                                </>
                            )}
                        </Nav>
                    </Navbar.Collapse>
                </Container>
            </Navbar>

            <Routes>
                <Route
                    path="/datasets/:datasetId/classifications/:classificationId/results/:resultId"
                    element={<ExplanationPage />}
                />
                <Route
                    path="/datasets/:datasetId/classifications_hotel/:classificationId/results/:resultId"
                    element={<ExplanationPageHotel />}
                />
                <Route
                    path="/datasets/:datasetId/classifications_pub/:classificationId/results/:resultId"
                    element={<ExplanationPagePubMedQA />}
                />
                 <Route
                    path="/datasets/:datasetId/classifications_ecqa/:classificationId/results/:resultId"
                    element={<ExplanationPageECQA />}
                />
                <Route
                    path="/datasets/:datasetId/classifications_snarks/:classificationId/results/:resultId"
                    element={<ExplanationPageSnarks />}
                />
                <Route
                    path="/predictions/:predictionId"
                    element={<ExplanationPageDashboard />}
                />
                <Route path="/" element={<PrivateRoute element={<HomePage />} />} />
                <Route path="/Dashboard" element={<PrivateRoute element={<Dashboard />} />} />
                <Route path="/login" element={<Login />} />
                <Route path="/register" element={<Register />} />
                <Route path="/datasets" element={<PrivateRoute element={<Datasets />} />} />
                <Route path="/settings" element={<PrivateRoute element={<Settings />} />} />
                <Route path="/dataset/:datasetId" element={<PrivateRoute element={<DatasetView />} />} />
                <Route path="/datasets/:datasetId/classifications_legal/:classificationId" element={<PrivateRoute element={<ClassificationDashboard />} />}/>
                <Route path="/datasets/:datasetId/classifications/:classificationId" element={<PrivateRoute element={<SentimentDashboard />} />}/>
                <Route path="/datasets/:datasetId/classifications_snarks/:classificationId" element={<PrivateRoute element={<SnarksDashboard />} />}/>
                <Route path="/datasets/:datasetId/classificationsp/:classificationId" element={<PrivateRoute element={<ClassificationDashboardPubMedQA />} />}/>
                <Route path="/datasets/:datasetId/classifications_ecqa/:classificationId" element={<PrivateRoute element={<ECQADashboard />} />}/>
                <Route path="/datasets/:datasetId/classifications_hotel/:classificationId" element={<PrivateRoute element={<HotelDashboard />} />}/>

            </Routes>
        </>
    );
}

function App() {
    return (
        <AuthProvider>
            <ProviderContextProvider>

            <Router>
                <AppContent />
            </Router>
            </ProviderContextProvider>

        </AuthProvider>
    );
}

export default App;