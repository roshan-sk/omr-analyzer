import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, tap } from 'rxjs';

@Injectable({
  providedIn: 'root',
})

export class AuthService {
  private apiUrl = 'http://localhost:5000';
  private cachedUser: any = null;

  constructor(private http: HttpClient) {}

  login(data: any) {
    console.log(data);
    return this.http.post(`${this.apiUrl}/login`, data, { withCredentials: true });
  }

  getCurrentUser() {
    if (this.cachedUser) {
      return new Observable((observer) => {
        observer.next(this.cachedUser);
        observer.complete();
      });
    }

    return this.http.get(`${this.apiUrl}/api/me`, { withCredentials: true }).pipe(
      tap((user: any) => {
        this.cachedUser = user;
      })
    );
  }

  clearUserCache() {
    this.cachedUser = null;
  }
}