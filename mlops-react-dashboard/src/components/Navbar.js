
import React, { useState } from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faBell, faCog, faEnvelopeOpen, faSearch, faSignOutAlt, faUserShield } from "@fortawesome/free-solid-svg-icons";
import { faUserCircle } from "@fortawesome/free-regular-svg-icons";
import { Row, Col, Nav, Form, Image, Navbar, Dropdown, Container, ListGroup, InputGroup } from '@themesberg/react-bootstrap';

import NOTIFICATIONS_DATA from "../data/notifications";
import Profile3 from "../assets/img/team/profile-picture-3.jpg";
import pipelines from "../data/pipelines";
import { faPiedPiper } from "@fortawesome/free-brands-svg-icons";
import swal from 'sweetalert';
const axios = require('axios');


export default class MyNavbar extends React.Component {
	  constructor(props) {
		super(props);
		this.state = {
			pipelines: [],
			selected_pipeline: ""
		}
	  
	  	axios.get('http://localhost:5000/pipelines', {}).then((res) => {
			var old_selected_pipeline = this.state.selected_pipeline
			this.setState({
				pipelines: res.data.pipelines,
				selected_pipeline: res.data.pipelines[0]
			});
			if (old_selected_pipeline!=res.data.pipelines[0]) {
				//this.props.history.push(window.location.pathname + window.location.hash));
				//console.log(window.location.pathname + window.location.hash)
			}
		}).catch((err) => {
			if (err.response && err.response.data && err.response.data.errorMessage) {
				swal({
					text: err.response.data.errorMessage,
					icon: "error",
					type: "error"
			});
		}
		});

		this.updatePipleline = this.updatePipleline.bind(this);
	}

	updatePipleline(e) {
		this.setState({selected_pipeline: e.target.innerText});

		//this.props.history.push(window.location.pathname + window.location.hash);
	}
  render() {

  return (
    <Navbar variant="dark" expanded className="ps-0 pe-2 pb-0">
      <Container fluid className="px-0">
        <div className="d-flex justify-content-between w-100">
          <div className="d-flex align-items-center">
		  	<Dropdown as={Nav.Item}>
              <Dropdown.Toggle as={Nav.Link} className="pt-1 px-0">
                <div className="media d-flex align-items-center">
                  <div className="media-body ms-2 text-dark align-items-center d-none d-lg-block">
                    <span className="mb-0 font-small fw-bold"><h2>Pipeline - {this.state.selected_pipeline}</h2></span>
                  </div>
                </div>
              </Dropdown.Toggle>
              <Dropdown.Menu className="user-dropdown dropdown-menu-right mt-2">
				{this.state.pipelines.map(p => 
					<Dropdown.Item className="fw-bold" key={p} onClick={this.updatePipleline}>
						<FontAwesomeIcon icon={faPiedPiper} className="me-2" /> {`${p}`}
			  		</Dropdown.Item>
				)}

                <Dropdown.Divider />

                <Dropdown.Item className="fw-bold">
                  <FontAwesomeIcon icon={faSignOutAlt} className="text-danger me-2" /> Logout
                </Dropdown.Item>
              </Dropdown.Menu>
            </Dropdown>
          </div>
          <Nav className="align-items-center">

            <Dropdown as={Nav.Item}>
              <Dropdown.Toggle as={Nav.Link} className="pt-1 px-0">
				{/* 
                <div className="media d-flex align-items-center">
                  <Image src={Profile3} className="user-avatar md-avatar rounded-circle" />
                  <div className="media-body ms-2 text-dark align-items-center d-none d-lg-block">
                    <span className="mb-0 font-small fw-bold">Bonnie Green</span>
                  </div>
                </div>
				*/}
              </Dropdown.Toggle>
              <Dropdown.Menu className="user-dropdown dropdown-menu-right mt-2">
                <Dropdown.Item className="fw-bold">
                  <FontAwesomeIcon icon={faUserCircle} className="me-2" /> My Profile
                </Dropdown.Item>
                <Dropdown.Item className="fw-bold">
                  <FontAwesomeIcon icon={faCog} className="me-2" /> Settings
                </Dropdown.Item>
                <Dropdown.Item className="fw-bold">
                  <FontAwesomeIcon icon={faEnvelopeOpen} className="me-2" /> Messages
                </Dropdown.Item>
                <Dropdown.Item className="fw-bold">
                  <FontAwesomeIcon icon={faUserShield} className="me-2" /> Support
                </Dropdown.Item>

                <Dropdown.Divider />

                <Dropdown.Item className="fw-bold">
                  <FontAwesomeIcon icon={faSignOutAlt} className="text-danger me-2" /> Logout
                </Dropdown.Item>
              </Dropdown.Menu>
            </Dropdown>
          </Nav>
        </div>
      </Container>
    </Navbar>
  );
  }
};
