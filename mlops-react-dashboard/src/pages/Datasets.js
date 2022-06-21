import React from "react";
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faCheck, faCog, faHome, faSearch } from '@fortawesome/free-solid-svg-icons';
import { Col, Row, Form, Button, ButtonGroup, Breadcrumb, InputGroup, Dropdown } from '@themesberg/react-bootstrap';
import { faCashRegister, faChartLine, faCloudUploadAlt, faPlus, faRocket, faTasks, faUserShield } from '@fortawesome/free-solid-svg-icons';

import { DatasetsTable } from "../components/Tables";
import swal from 'sweetalert';
const axios = require('axios');

export default class Datasets extends React.Component {
	constructor(props) {
		super(props);
		this.state = {
			datasets: []
		}
		axios.get('http://localhost:5000/datasets', {}).then((res) => {
			//this.props.history.push('/dashboard');
			console.log(res.data.datasets)
			this.setState({
				datasets: res.data.datasets
			});
		}).catch((err) => {
			if (err.response && err.response.data && err.response.data.errorMessage) {
				swal({
					text: err.response.data.errorMessage,
					icon: "error",
					type: "error"
			});
		}
		});

	}
  render() {
	return (
		<div>
		  <div className="d-flex justify-content-between flex-wrap flex-md-nowrap align-items-center py-4">
			<div className="d-block mb-4 mb-md-0">
			  <Breadcrumb className="d-none d-md-inline-block" listProps={{ className: "breadcrumb-dark breadcrumb-transparent" }}>
				<Breadcrumb.Item><FontAwesomeIcon icon={faHome} /></Breadcrumb.Item>
				<Breadcrumb.Item>ML Ops</Breadcrumb.Item>
				<Breadcrumb.Item active>Datasets</Breadcrumb.Item>
			  </Breadcrumb>
			  <h4>Datasets</h4>
			  <p className="mb-0">Your web analytics dashboard template.</p>
			</div>
			<div className="btn-toolbar mb-2 mb-md-0">
			  <ButtonGroup>
				<Button variant="outline-primary" size="sm">Share</Button>
				<Button variant="outline-primary" size="sm">Export</Button>
			  </ButtonGroup>
			</div>
		  </div>
	
		  <div className="table-settings mb-4">
			<Row className="justify-content-between align-items-center">
			  <Col xs={8} md={6} lg={3} xl={4}>
				<InputGroup>
				  <InputGroup.Text>
					<FontAwesomeIcon icon={faSearch} />
				  </InputGroup.Text>
				  <Form.Control type="text" placeholder="Search" />
				</InputGroup>
			  </Col>
			  <Col xs={4} md={2} xl={1} className="ps-md-0 text-end">
				<Dropdown as={ButtonGroup}>
				  <Dropdown.Toggle split as={Button} variant="link" className="text-dark m-0 p-0">
					<span className="icon icon-sm icon-gray">
					  <FontAwesomeIcon icon={faCog} />
					</span>
				  </Dropdown.Toggle>
				  <Dropdown.Menu className="dropdown-menu-xs dropdown-menu-right">
					<Dropdown.Item className="fw-bold text-dark">Show</Dropdown.Item>
					<Dropdown.Item className="d-flex fw-bold">
					  10 <span className="icon icon-small ms-auto"><FontAwesomeIcon icon={faCheck} /></span>
					</Dropdown.Item>
					<Dropdown.Item className="fw-bold">20</Dropdown.Item>
					<Dropdown.Item className="fw-bold">30</Dropdown.Item>
				  </Dropdown.Menu>
				</Dropdown>
			  </Col>
			</Row>
		  </div>
	
		  <div className="table-settings mb-4">
			  <Button variant="primary" size="sm" className="me-2">
				<FontAwesomeIcon icon={faPlus} className="me-2" />New Dataset
			  </Button>
		  </div>
		  
	
		  <DatasetsTable datasets={this.state.datasets} />
		</div>
	  );
  }
};
