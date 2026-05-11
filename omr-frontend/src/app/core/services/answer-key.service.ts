import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Injectable({
  providedIn: 'root'
})

export class AnswerKeyService {

  private apiUrl = 'http://localhost:5000/api';

  constructor(private http: HttpClient) {}

  getAnswerKey(level: string) {
    return this.http.get(`${this.apiUrl}/get_answer_key/${level}`);
  }

  saveAnswerKey(data: any) {
    return this.http.post(
      `${this.apiUrl}/save_answer_key`,
      data
    );
  }
}